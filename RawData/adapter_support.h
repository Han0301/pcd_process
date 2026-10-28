#pragma once

#include "metabag_io.h"
#include <functional>
#include <limits>
#include <iomanip>
#include <sstream>
#include <cstdlib>

struct OfflineSource
{
    std::string source_file;
    std::uint64_t source_frame = 0;
    std::uint64_t frame_id = 0;
    std::uint32_t raw_points = 0;
    std::uint64_t source_header_ns = 0;
    std::uint64_t source_timebase_ns = 0;
    std::uint32_t source_max_offset_ns = 0;
};

inline std::int64_t offline_ns(double stamp)
{
    return static_cast<std::int64_t>(std::llround(stamp * 1e9));
}

inline void read_ros_header(
    offline_metabag::Cursor& cursor,
    std_msgs::msg::Header& header)
{
    cursor.get<std::uint32_t>();
    header.stamp.sec = static_cast<std::int32_t>(cursor.get<std::uint32_t>());
    header.stamp.nanosec = cursor.get<std::uint32_t>();
    header.frame_id = cursor.string();
}

inline sensor_msgs::msg::Imu::UniquePtr decode_imu(const offline_metabag::Bytes& data)
{
    auto message = std::make_unique<sensor_msgs::msg::Imu>();
    offline_metabag::Cursor cursor{data.data(), data.size()};
    read_ros_header(cursor, message->header);
    message->orientation.x = cursor.get<double>();
    message->orientation.y = cursor.get<double>();
    message->orientation.z = cursor.get<double>();
    message->orientation.w = cursor.get<double>();
    for (auto& value : message->orientation_covariance)
    {
        value = cursor.get<double>();
    }
    message->angular_velocity.x = cursor.get<double>();
    message->angular_velocity.y = cursor.get<double>();
    message->angular_velocity.z = cursor.get<double>();
    for (auto& value : message->angular_velocity_covariance)
    {
        value = cursor.get<double>();
    }
    message->linear_acceleration.x = cursor.get<double>();
    message->linear_acceleration.y = cursor.get<double>();
    message->linear_acceleration.z = cursor.get<double>();
    for (auto& value : message->linear_acceleration_covariance)
    {
        value = cursor.get<double>();
    }
    if (cursor.remaining)
    {
        throw std::runtime_error("Unexpected IMU payload tail");
    }
    return message;
}

inline livox_ros_driver2::msg::CustomMsg::UniquePtr decode_livox(const offline_metabag::Bytes& data)
{
    auto message = std::make_unique<livox_ros_driver2::msg::CustomMsg>();
    offline_metabag::Cursor cursor{data.data(), data.size()};
    read_ros_header(cursor, message->header);
    message->timebase = cursor.get<std::uint64_t>();
    message->point_num = cursor.get<std::uint32_t>();
    message->lidar_id = cursor.get<std::uint8_t>();
    for (auto& value : message->rsvd)
    {
        value = cursor.get<std::uint8_t>();
    }
    const auto point_count = cursor.get<std::uint32_t>();
    if (point_count != message->point_num || point_count > 1000000 || cursor.remaining != point_count * 19ULL)
    {
        throw std::runtime_error("Invalid Livox point array");
    }
    message->points.resize(point_count);
    for (auto& point : message->points)
    {
        point.offset_time = cursor.get<std::uint32_t>();
        point.x = cursor.get<float>();
        point.y = cursor.get<float>();
        point.z = cursor.get<float>();
        point.reflectivity = cursor.get<std::uint8_t>();
        point.tag = cursor.get<std::uint8_t>();
        point.line = cursor.get<std::uint8_t>();
    }
    return message;
}

class OfflineOutput
{
public:
    OfflineOutput(const std::string& path, double frame_period)
        : path_(path), frame_period_(frame_period)
    {
        std::filesystem::create_directories(path_ / "frames");
        trajectory_.open(path_ / "trajectory.csv");
        statuses_.open(path_ / "frame_status.csv");
        frames_.open(path_ / "frames.csv");
        trajectory_ << "frame_id,source_file,source_frame,stamp_ns,scan_start_ns,scan_end_ns,stamp,scan_start,scan_end,x,y,z,qx,qy,qz,qw,gx,gy,gz,input_points,downsampled_points,effective_points,residual_mean";
        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                trajectory_ << ",m" << row << column;
            }
        }
        trajectory_ << ",source_header_ns,source_timebase_ns,source_max_offset_ns\n" << std::setprecision(17);
        statuses_ << "frame_id,source_file,source_frame,scan_start,scan_end,status,raw_points,preprocessed_points,deskewed_points,downsampled_points,effective_points,residual_mean\n" << std::setprecision(17);
        frames_ << "file,offset_file,frame_id,stamp_ns,stamp,scan_start,scan_end,points,x,y,z,qx,qy,qz,qw,source_header_ns,source_timebase_ns,source_max_offset_ns\n" << std::setprecision(17);
    }

    void record(
        const std::string& status,
        const OfflineSource& source,
        const MeasureGroup& measures,
        const state_ikfom& state,
        const PointCloudXYZI& cloud,
        int downsampled,
        int effective,
        double residual)
    {
        const bool success = status == "optimized" && effective >= 5 && state.pos.allFinite();
        statuses_ << source.frame_id << ',' << source.source_file << ',' << source.source_frame << ','
            << measures.lidar_beg_time << ',' << measures.lidar_end_time << ','
            << (status == "optimized" && !success ? "invalid_measurement" : status) << ','
            << source.raw_points << ',' << measures.lidar->size() << ',' << cloud.size() << ','
            << downsampled << ',' << effective << ',' << residual << '\n';
        ++consumed_frames;
        if (!success)
        {
            ++unsuccessful_frames;
            return;
        }
        ++successful_frames;
        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
        matrix.block<3, 3>(0, 0) = state.rot.toRotationMatrix() * state.offset_R_L_I.toRotationMatrix();
        matrix.block<3, 1>(0, 3) = state.pos + state.rot * state.offset_T_L_I;
        const Eigen::Quaterniond quaternion(matrix.block<3, 3>(0, 0));
        const double stamp = measures.lidar_end_time;
        std::string physical_failure;
        const double state_speed = state.vel.norm();
        const bool valid_rotation = matrix.allFinite() &&
            std::abs(matrix.block<3, 3>(0, 0).determinant() - 1) < 1e-6;
        if (!valid_rotation || !state.vel.allFinite())
        {
            physical_failure = "nonfinite_or_invalid_rotation";
        }
        if (matrix.block<3, 1>(0, 3).norm() > 200)
        {
            physical_failure = "position_exceeds_200m_indoor_envelope";
        }
        high_speed_frames_ = state_speed > 3 ? high_speed_frames_ + 1 : 0;
        if (high_speed_frames_ >= 3)
        {
            physical_failure = "state_speed_exceeds_3mps_for_3_frames";
        }
        double step_speed = 0;
        double angular_step = 0;
        if (has_previous_pose_)
        {
            const double dt = stamp - previous_stamp_;
            step_speed = (matrix.block<3, 1>(0, 3) - previous_pose_.block<3, 1>(0, 3)).norm() / std::max(dt, 1e-6);
            angular_step = Eigen::AngleAxisd(matrix.block<3, 3>(0, 0) * previous_pose_.block<3, 3>(0, 0).transpose()).angle() * 180 / M_PI;
            if (dt <= 0 || step_speed > 5)
            {
                physical_failure = "nonpositive_time_or_step_speed_exceeds_5mps";
            }
            if (angular_step > 100)
            {
                physical_failure = "angular_step_exceeds_100deg";
            }
        }
        if (!physical_failure.empty())
        {
            std::ofstream failure(path_ / "run_abort.json");
            failure << std::setprecision(17) << "{\n  \"reason\": \"" << physical_failure
                << "\",\n  \"frame_id\": " << source.frame_id
                << ",\n  \"source_file\": \"" << source.source_file
                << "\",\n  \"source_frame\": " << source.source_frame
                << ",\n  \"stamp\": " << stamp
                << ",\n  \"state_speed_mps\": " << state_speed
                << ",\n  \"step_speed_mps\": " << step_speed
                << ",\n  \"angular_step_deg\": " << angular_step
                << ",\n  \"point_deletion_performed\": false\n}\n";
            throw std::runtime_error("Physical divergence fuse: " + physical_failure);
        }
        previous_pose_ = matrix;
        previous_stamp_ = stamp;
        has_previous_pose_ = true;
        trajectory_ << source.frame_id << ',' << source.source_file << ',' << source.source_frame << ','
            << offline_ns(stamp) << ',' << offline_ns(measures.lidar_beg_time) << ',' << offline_ns(stamp) << ','
            << stamp << ',' << measures.lidar_beg_time << ',' << stamp << ','
            << matrix(0, 3) << ',' << matrix(1, 3) << ',' << matrix(2, 3) << ','
            << quaternion.x() << ',' << quaternion.y() << ',' << quaternion.z() << ',' << quaternion.w() << ','
            << state.grav[0] << ',' << state.grav[1] << ',' << state.grav[2] << ','
            << measures.lidar->size() << ',' << downsampled << ',' << effective << ',' << residual;
        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                trajectory_ << ',' << matrix(row, column);
            }
        }
        trajectory_ << ',' << source.source_header_ns << ',' << source.source_timebase_ns << ','
            << source.source_max_offset_ns << '\n';
        if (!has_export_origin_)
        {
            export_origin_ = stamp;
            has_export_origin_ = true;
        }
        const auto sample_id = frame_period_ > 0 ?
            static_cast<std::int64_t>(std::floor((stamp - export_origin_ + 0.005) / frame_period_)) :
            static_cast<std::int64_t>(successful_frames);
        if (sample_id <= last_sample_id_)
        {
            return;
        }
        last_sample_id_ = sample_id;
        const std::string name = std::to_string(offline_ns(stamp));
        const std::string file_name = "frames/" + name + ".bin";
        const std::string offset_name = "frames/" + name + "_offset_ms.bin";
        std::ofstream points(path_ / file_name, std::ios::binary);
        std::ofstream offsets(path_ / offset_name, std::ios::binary);
        for (const auto& point : cloud)
        {
            const float values[4] = {point.x, point.y, point.z, point.intensity};
            points.write(reinterpret_cast<const char*>(values), sizeof(values));
            offsets.write(reinterpret_cast<const char*>(&point.curvature), sizeof(float));
        }
        if (!points || !offsets)
        {
            throw std::runtime_error("Writing deskewed frame failed");
        }
        frames_ << file_name << ',' << offset_name << ',' << source.frame_id << ',' << offline_ns(stamp) << ','
            << stamp << ',' << measures.lidar_beg_time << ',' << stamp << ',' << cloud.size() << ','
            << matrix(0, 3) << ',' << matrix(1, 3) << ',' << matrix(2, 3) << ','
            << quaternion.x() << ',' << quaternion.y() << ',' << quaternion.z() << ',' << quaternion.w() << ','
            << source.source_header_ns << ',' << source.source_timebase_ns << ',' << source.source_max_offset_ns << '\n';
        ++saved_frames;
    }

    std::uint64_t successful_frames = 0;
    std::uint64_t unsuccessful_frames = 0;
    std::uint64_t consumed_frames = 0;
    std::uint64_t saved_frames = 0;

private:
    std::filesystem::path path_;
    double frame_period_;
    double export_origin_ = 0;
    bool has_export_origin_ = false;
    std::int64_t last_sample_id_ = -1;
    Eigen::Matrix4d previous_pose_ = Eigen::Matrix4d::Identity();
    double previous_stamp_ = 0;
    bool has_previous_pose_ = false;
    int high_speed_frames_ = 0;
    std::ofstream trajectory_;
    std::ofstream statuses_;
    std::ofstream frames_;
};

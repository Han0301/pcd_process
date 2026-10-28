#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace offline_metabag
{
using Bytes = std::vector<std::uint8_t>;
using Fields = std::map<std::string, Bytes>;

inline std::uint32_t u32(const std::uint8_t* p)
{
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

struct Cursor
{
    const std::uint8_t* p;
    std::size_t remaining;

    template<typename T> T get()
    {
        if (remaining < sizeof(T))
        {
            throw std::runtime_error("Truncated ROS1 message");
        }
        T value;
        std::memcpy(&value, p, sizeof(T));
        p += sizeof(T);
        remaining -= sizeof(T);
        return value;
    }

    std::string string()
    {
        const auto length = get<std::uint32_t>();
        if (length > remaining)
        {
            throw std::runtime_error("Truncated ROS1 string");
        }
        std::string value(reinterpret_cast<const char*>(p), length);
        p += length;
        remaining -= length;
        return value;
    }
};

inline Fields fields(const Bytes& header)
{
    Fields result;
    Cursor cursor{header.data(), header.size()};
    while (cursor.remaining)
    {
        const auto length = cursor.get<std::uint32_t>();
        if (length > cursor.remaining)
        {
            throw std::runtime_error("Truncated METABAG header field");
        }
        const auto* end = cursor.p + length;
        const auto* equals = std::find(cursor.p, end, '=');
        if (equals == end)
        {
            throw std::runtime_error("Malformed METABAG header field");
        }
        result[std::string(reinterpret_cast<const char*>(cursor.p), equals - cursor.p)] =
            Bytes(equals + 1, end);
        cursor.p = end;
        cursor.remaining -= length;
    }
    return result;
}

inline std::string text(const Fields& fields, const std::string& name)
{
    const auto found = fields.find(name);
    if (found == fields.end())
    {
        return {};
    }
    return std::string(found->second.begin(), found->second.end());
}

inline std::uint32_t integer(const Fields& fields, const std::string& name)
{
    const auto found = fields.find(name);
    if (found == fields.end() || found->second.size() != 4)
    {
        throw std::runtime_error("Missing integer METABAG header: " + name);
    }
    return u32(found->second.data());
}

struct Message
{
    std::uint32_t connection = 0;
    std::uint64_t record_ns = 0;
    Bytes data;
    std::string source_file;
};

class Reader
{
public:
    explicit Reader(const std::string& directory)
    {
        for (const auto& item : std::filesystem::directory_iterator(directory))
        {
            const std::string name = item.path().filename().string();
            if (name.rfind("data_", 0) == 0 && name.size() > 5 &&
                std::all_of(name.begin() + 5, name.end(), [](char c)
                {
                    return c >= '0' && c <= '9';
                }))
            {
                files_.emplace_back(std::stoi(name.substr(5)), item.path());
            }
        }
        std::sort(files_.begin(), files_.end());
        if (files_.empty())
        {
            throw std::runtime_error("No METABAG data_N files");
        }
    }

    bool next(Message& message)
    {
        while (true)
        {
            if (chunk_position_ < chunk_.size())
            {
                Cursor cursor{chunk_.data() + chunk_position_, chunk_.size() - chunk_position_};
                const auto header_size = cursor.get<std::uint32_t>();
                if (header_size > cursor.remaining)
                {
                    throw std::runtime_error("Truncated chunk record header");
                }
                const Bytes header(cursor.p, cursor.p + header_size);
                cursor.p += header_size;
                cursor.remaining -= header_size;
                const auto data_size = cursor.get<std::uint32_t>();
                if (data_size > cursor.remaining)
                {
                    throw std::runtime_error("Truncated chunk record data");
                }
                const Fields values = fields(header);
                const auto op = values.at("op").at(0);
                chunk_position_ = static_cast<std::size_t>(cursor.p - chunk_.data()) + data_size;
                if (op == 0x07)
                {
                    connection_types_[integer(values, "conn")] =
                        text(fields(Bytes(cursor.p, cursor.p + data_size)), "type");
                }
                if (op != 0x02)
                {
                    continue;
                }
                const auto source_connection = integer(values, "conn");
                const auto connection_type = connection_types_.find(source_connection);
                if (connection_type == connection_types_.end())
                {
                    throw std::runtime_error("Message before connection declaration");
                }
                if (connection_type->second == "sensor_msgs/Imu")
                {
                    message.connection = 0;
                }
                else if (connection_type->second.find("CustomMsg") != std::string::npos)
                {
                    message.connection = 1;
                }
                else
                {
                    continue;
                }
                const auto& timestamp = values.at("time");
                if (timestamp.size() != 8)
                {
                    throw std::runtime_error("Invalid bag record timestamp");
                }
                message.record_ns = static_cast<std::uint64_t>(u32(timestamp.data())) * 1000000000ULL +
                    u32(timestamp.data() + 4);
                message.source_file = files_.at(file_index_ - 1).second.filename().string();
                message.data.assign(cursor.p, cursor.p + data_size);
                return true;
            }

            if (!input_.is_open() || input_.peek() == std::char_traits<char>::eof())
            {
                if (file_index_ == files_.size())
                {
                    return false;
                }
                input_.close();
                input_.open(files_[file_index_++].second, std::ios::binary);
                connection_types_.clear();
                std::string magic;
                std::getline(input_, magic);
                if (!input_ || magic != "#METABAG V2.0")
                {
                    throw std::runtime_error("Invalid METABAG magic");
                }
            }

            std::uint32_t header_size = 0;
            input_.read(reinterpret_cast<char*>(&header_size), 4);
            if (!input_ || header_size > 16777216)
            {
                throw std::runtime_error("Invalid outer record header");
            }
            Bytes header(header_size);
            input_.read(reinterpret_cast<char*>(header.data()), header.size());
            std::uint32_t data_size = 0;
            input_.read(reinterpret_cast<char*>(&data_size), 4);
            if (!input_ || data_size > 1073741824)
            {
                throw std::runtime_error("Invalid outer record payload");
            }
            const Fields values = fields(header);
            if (values.at("op").at(0) != 0x05)
            {
                input_.seekg(data_size, std::ios::cur);
                continue;
            }
            if (text(values, "compression") != "none")
            {
                throw std::runtime_error("Compressed METABAG is unsupported");
            }
            chunk_.resize(data_size);
            input_.read(reinterpret_cast<char*>(chunk_.data()), chunk_.size());
            if (!input_)
            {
                throw std::runtime_error("Truncated METABAG chunk");
            }
            chunk_position_ = 0;
        }
    }

private:
    std::vector<std::pair<int, std::filesystem::path>> files_;
    std::size_t file_index_ = 0;
    std::ifstream input_;
    Bytes chunk_;
    std::size_t chunk_position_ = 0;
    std::map<std::uint32_t, std::string> connection_types_;
};
}

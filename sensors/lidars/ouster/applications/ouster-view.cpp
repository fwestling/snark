#include <unistd.h>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdlib>

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <comma/application/command_line_options.h>

#include "../os1.h"
#include "../os1_packet.h"
#include "ouster_view/viz.h"

namespace OS1 = ouster::OS1;
namespace viz = ouster::viz;

/**
 * Generate a pre-computed matrix of unit vectors pointing radially outwards,
 * for easily computing a point cloud from a lidar scan
 */
std::vector<double> make_xyz_lut(const viz::SensorSpecifics& ss) {
    int n = ss.H * ss.col_per_rev;
    std::vector<double> xyz = std::vector<double>(3 * n, 0);
    double* x(xyz.data());
    double* y(xyz.data() + n);
    double* z(xyz.data() + 2 * n);

    for (int icol = 0; icol < ss.col_per_rev; icol++) {
        for (int ipx = 0; ipx < OS1::pixels_per_column; ipx++) {
            double h_angle_0 =
                (2.0 * M_PI * (double)icol) / ((double)ss.col_per_rev);
            double h_angle = std::sin((double)ss.beam_azimuth_angles.at(ipx) *
                                      2 * M_PI / 360.0) +
                             h_angle_0;

            x[(ipx * ss.col_per_rev) + icol] =
                -std::cos((double)ss.beam_altitude_angles.at(ipx) * 2 * M_PI /
                          360.0) *
                std::cos(h_angle);
            y[(ipx * ss.col_per_rev) + icol] =
                std::cos((double)ss.beam_altitude_angles.at(ipx) * 2 * M_PI /
                         360.0) *
                std::sin(h_angle);
            z[(ipx * ss.col_per_rev) + icol] = std::sin(
                (double)ss.beam_altitude_angles.at(ipx) * 2 * M_PI / 360.0);
        }
    }
    return xyz;
}

/**
 * Add a LiDAR UDP column to the lidar scan
 */
int add_col_to_lidar_scan(const uint8_t* col_buf,
                          std::unique_ptr<ouster::LidarScan>& lidar_scan) {
    const int ticks = OS1::col_h_encoder_count(col_buf);
    // drop packets with invalid encoder counts
    if (ticks < 0 or ticks > OS1::encoder_ticks_per_rev) {
        return 0;
    }
    const int azimuth = ticks * lidar_scan->W / OS1::encoder_ticks_per_rev;

    for (size_t row = 0; row < OS1::pixels_per_column; row++) {
        const uint8_t* px_buf = OS1::nth_px(row, col_buf);
        size_t index = row * lidar_scan->W + azimuth;
        lidar_scan->range.at(index) = OS1::px_range(px_buf);
        lidar_scan->intensity.at(index) = OS1::px_signal_photons(px_buf);
        lidar_scan->reflectivity.at(index) = OS1::px_reflectivity(px_buf);
        lidar_scan->noise.at(index) = OS1::px_noise_photons(px_buf);
    }
    return azimuth;
}

/**
 * Add a LiDAR UDP packet to the Point OS1 cloud
 */
template <typename F>
void add_packet_to_lidar_scan(const uint8_t* buf, int& last_azimuth,
                              std::unique_ptr<ouster::LidarScan>& lidar_scan,
                              F&& f) {
    for (int icol = 0; icol < OS1::columns_per_buffer; icol++) {
        const uint8_t* col_buf = OS1::nth_col(icol, buf);
        // individual column gets dropped if the column is invalid
        if (!OS1::col_valid(col_buf)) continue;
        int azimuth =
            (add_col_to_lidar_scan(col_buf, lidar_scan) + 1) % (lidar_scan->W);

        if (azimuth < last_azimuth) f();
        last_azimuth = azimuth;
    }
}

void usage( bool verbose = false )
{
    std::cerr << "\nvisualise Ouster OS-1 lidar data on port 7502";
    std::cerr << "\n";
    std::cerr << "\nusage: ouster-view [<options>]";
    std::cerr << "\n";
    std::cerr << "\noptions:";
    std::cerr << "\n    --help,-h;               display this help message and exit";
    std::cerr << "\n    --verbose,-v;            more output";
    std::cerr << "\n    --host=[<hostname>];     hostname or ip address of lidar unit";
    std::cerr << "\n    --intensity-scale,-i;    intensity image scaling factor";
    std::cerr << "\n    --noise-scale,-n;        noise image scaling factor";
    std::cerr << "\n    --range-scale,-r;        range image scaling factor";
    std::cerr << "\n    --udp-dest=[<ip addr>];  ip address of data destination";
    std::cerr << "\n";
    std::cerr << "\n    if both --host and --udp-dest are set then the lidar unit at --host";
    std::cerr << "\n    is initialized to transmit to --udp-dest";
    std::cerr << "\n";
    if( verbose )
    {
        std::cerr << "\nkey bindings:";
        std::cerr << "\n";
        std::cerr << "\n      o | Increase point size";
        std::cerr << "\n      p | Decrease point size";
        std::cerr << "\n      i | Colour 3D points by intensity";
        std::cerr << "\n      z | Colour 3D points by z-height";
        std::cerr << "\n      Z | Colour 3D points by z-height plus intensity";
        std::cerr << "\n      r | Colour 3D points by range";
        std::cerr << "\n      c | Cycle colour for accumulated 3D points and range image";
        std::cerr << "\n      C | Cycle colour for latest 3D points";
        std::cerr << "\n      v | Toggle colour cycling in range image";
        std::cerr << "\n      n | Display noise image from the sensor|";
        std::cerr << "\n      a | Rotate camera to show 3D points in top down view";
        std::cerr << "\n      f | Rotate camera to show 3D points in front view";
        std::cerr << "\n   left | Rotate camera left";
        std::cerr << "\n  right | Rotate camera right";
        std::cerr << "\n     up | Rotate camera up";
        std::cerr << "\n   down | Rotate camera down";
        std::cerr << "\n      + | Move camera closer";
        std::cerr << "\n      - | Move camera farther";
        std::cerr << "\n      0 | Toggle parallel projection";
        std::cerr << "\n      d | Decrease range and intensity window height";
        std::cerr << "\n";
        std::cerr << "\nmouse control:";
        std::cerr << "\n";
        std::cerr << "\n    Click and drag rotates the view";
        std::cerr << "\n    Middle click and drag pans the view";
        std::cerr << "\n    Scroll adjusts camera distance";
        std::cerr << "\n";
        std::cerr << "\nmodes:";
        std::cerr << "\n";
        std::cerr << "\n    * Cycle range image colour: colour is repeated every few meters";
        std::cerr << "\n    * Parallel projection: renders 3D point cloud without perspective distortion";
        std::cerr << "\n    * Colour mode: possible values:";
        std::cerr << "\n          z-height, intensity, range, and z-height plus intensity";
        std::cerr << "\n          z-height plus intensity by default";
        std::cerr << "\n";
    }
    else
    {
        std::cerr << "\nrun --help --verbose for details on key bindings\n";
    }
    std::cerr << "\nexamples:";
    std::cerr << "\n    (view data directly from the lidar)";
    std::cerr << "\n    ouster-view --host os1-991832000987.local --udp-dest 192.168.1.1";
    std::cerr << "\n";
    std::cerr << "\n    (in two terminals, start visualiser then feed it some data)";
    std::cerr << "\n    ouster-view";
    std::cerr << "\n    cat *.bin | csv-eval --fields t --binary ul,12600ub \"t=t/1000\" | csv-play --binary t,12600ub | socat -b12608 -u - udp:localhost:7502";
    std::cerr << "\n" << std::endl;
}

int main(int argc, char** argv)
{
    viz::UserConfig uc;
    viz::SensorSpecifics ss = { 1024, OS1::pixels_per_column,
                                std::vector<float>( OS1::beam_azimuth_angles.begin()
                                                  , OS1::beam_azimuth_angles.end()),
                                std::vector<float>( OS1::beam_altitude_angles.begin()
                                                  , OS1::beam_altitude_angles.end())
                              };
    try
    {
        comma::command_line_options options( argc, argv, usage );

        uc.intensity_scale = options.value< double >( "--intensity-scale,-i", 1.0 );
        uc.noise_scale = options.value< double >( "--noise-scale,-n", 1.0 );
        uc.range_scale = options.value< double >( "--range-scale,-n", 1.0 );

        std::string lidar_hostname = options.value< std::string >( "--host", "" );
        std::string udp_dest = options.value< std::string >( "--udp-dest", "" );
        if( lidar_hostname.empty() != udp_dest.empty() ) { COMMA_THROW( comma::exception, "if either --host or --udp-dest is set, both must be set" ); }

        std::shared_ptr< OS1::client > cli;
        cli = OS1::init_client( lidar_hostname, udp_dest, 7502, 7503 );

        comma::verbose << "tables " << ( OS1::tables_initialized ? "" : "not " ) << "initialized" << std::endl;

        uint8_t lidar_buf[OS1::lidar_packet_bytes + 1];
        uint8_t imu_buf[OS1::imu_packet_bytes + 1];

        auto ls_poll = std::unique_ptr<ouster::LidarScan>( new ouster::LidarScan( ss.col_per_rev, ss.H ));

        int ls_counter = ss.col_per_rev;

        std::vector<double> xyz_lut = make_xyz_lut(ss);
        auto vh = viz::init_viz(xyz_lut, uc, ss);

        // Use to signal termination
        std::atomic_bool end_program{false};

        // Start render loop
        std::thread render([&] { viz::run_viz(*vh); end_program = true; });

        // Poll the client for data and add to our lidar scan
        while (!end_program)
        {
            OS1::client_state st = OS1::poll_client(*cli);

            if (st & OS1::client_state::ERROR)
            {
                std::cerr << "Client returned error state" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (st & OS1::client_state::LIDAR_DATA)
            {
                if (OS1::read_lidar_packet(*cli, lidar_buf))
                {
                    add_packet_to_lidar_scan( lidar_buf, ls_counter, ls_poll
                                            , [&] { viz::update_poll(*vh, ls_poll); });
                }
            }
            if (st & OS1::client_state::IMU_DATA) { OS1::read_imu_packet(*cli, imu_buf); }
        }

        // clean up
        render.join();
        return 0;
    }
    catch( std::exception& ex ) { std::cerr << "ouster-view: " << ex.what() << std::endl; }
    catch( ... ) { std::cerr << "ouster-view: unknown exception" << std::endl; }
    return 1;
}

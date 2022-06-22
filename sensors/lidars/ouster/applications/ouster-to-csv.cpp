// Copyright (c) 2019 The University of Sydney

#include "../config.h"
#include "../packet.h"
#include "../traits.h"
#include "../types.h"
#include <comma/application/command_line_options.h>
#include <comma/csv/stream.h>
#include <comma/name_value/serialize.h>
#include <boost/date_time/posix_time/posix_time_types.hpp>

static const std::string default_config( "config.json:ouster" );

static void bash_completion( unsigned int const ac, char const * const * av )
{
    static const char* completion_options =
        " --help -h --verbose -v"
        " --config --output-fields --output-format --output-frame"
        " lidar imu"
        ;

    std::cout << completion_options << std::endl;
    exit( 0 );
}

void usage( bool verbose )
{
    std::cerr << "\nconvert raw Ouster OS-1 lidar data";
    std::cerr << "\n";
    std::cerr << "\nusage: cat <raw-data> | ouster-to-csv <lidar|imu> [<options>]";
    std::cerr << "\n";
    std::cerr << "\noptions:";
    std::cerr << "\n    --help,-h:             display this help message and exit";
    std::cerr << "\n    --verbose,-v:          more output";
    std::cerr << "\n    --config=<file:path>:  default: " << default_config;
    std::cerr << "\n    --output-fields:       list output fields and exit";
    std::cerr << "\n    --output-format:       list output format and exit";
    std::cerr << "\n    --output-frame:        output frame offset as x,y,z,r,p,y";
    std::cerr << "\n";
    std::cerr << "\n    for any particular device the config can be generated by";
    std::cerr << "\n    ouster-cat config --device <address>";
    std::cerr << "\n";
    std::cerr << "\nunits:";
    std::cerr << "\n    raw Ouster data is converted to regular SI units. In particular,";
    std::cerr << "\n    timestamps are ISO, range is in metres, and angular acceleration is rad/s.";
    std::cerr << "\n";
    std::cerr << "\nissues:";
    std::cerr << "\n    minimum range appears to be 700mm";
    std::cerr << "\n";
    if( verbose )
    {
        std::cerr << "\nexamples:";
        std::cerr << "\n    --- save fields and format";
        std::cerr << "\n    fields=$( ouster-to-csv lidar --output-fields )";
        std::cerr << "\n    format=$( ouster-to-csv lidar --output-format )";
        std::cerr << "\n";
        std::cerr << "\n    --- view live data ---";
        std::cerr << "\n    ouster-cat lidar --config config.json:ouster | ouster-to-csv lidar \\";
        std::cerr << "\n        | points-to-cartesian --append --fields $fields --binary $format \\";
        std::cerr << "\n        | view-points --fields $fields,x,y,z --binary $format,3d --z-is-up";
        std::cerr << "\n";
        std::cerr << "\n    --- point cloud ---";
        std::cerr << "\n    cat *.bin | ouster-to-csv lidar \\";
        std::cerr << "\n        | points-to-cartesian --append --fields $fields --binary $format \\";
        std::cerr << "\n        | csv-play --binary $format,3d \\";
        std::cerr << "\n        | view-points --fields $fields,x,y,z --binary $format,3d --z-is-up";
        std::cerr << "\n";
        std::cerr << "\n    --- image ---";
        std::cerr << "\n    data_field=ambient   # or signal or reflectivity";
        std::cerr << "\n    cat *.bin | ouster-to-csv lidar \\";
        std::cerr << "\n        | csv-select --fields=block --binary=$format --greater=0 --sorted \\";
        std::cerr << "\n        | csv-eval --fields=$fields --binary=$format \"bearing=bearing%(2*pi)\" \\";
        std::cerr << "\n        | csv-sort --fields=$fields --binary=$format --order=elevation,bearing \\";
        std::cerr << "\n        | csv-shuffle --fields $fields --binary $format --output $data_field \\";
        std::cerr << "\n        | cv-cat --input=\"rows=64;cols=1024;no-header;type=CV_16UC1\" \\";
        std::cerr << "\n                 \"flip;brightness=60;resize=1.0,2.0;view;null\"";
        std::cerr << "\n";
        std::cerr << "\n    --- time ---";
        std::cerr << "\n    usually ptp synchronized time will be used (timestamp_mode set to";
        std::cerr << "\n    TIME_FROM_PTP_1588) but if only time from power-on has been recorded";
        std::cerr << "\n    (timestamp_mode=TIME_FROM_INTERNAL_OSC) then a log file can be roughly";
        std::cerr << "\n    corrected with:";
        std::cerr << "\n    start=$( basename $( ls *.bin | head -1 ) | csv-time --to seconds )";
        std::cerr << "\n    cat *.bin | ouster-to-csv lidar | csv-time-delay --binary $format $start";
        std::cerr << "\n";
        std::cerr << "\n    --- intrinsic calibration ---";
        std::cerr << "\n    frame=$( ouster-to-csv lidar --output-frame )";
        std::cerr << "\n    cat *.bin | ouster-to-csv lidar \\";
        std::cerr << "\n        | points-to-cartesian --append --fields $fields --binary $format \\";
        std::cerr << "\n        | points-frame --from $frame --fields $fields,x,y,z --binary $format,3d \\";
        std::cerr << "\n        | view-points --fields $fields,x,y,z --binary $format,3d";
        std::cerr << "\n";
        std::cerr << "\n    --- live imu ---";
        std::cerr << "\n    fields=$( ouster-to-csv imu --output-fields )";
        std::cerr << "\n    format=$( ouster-to-csv imu --output-format )";
        std::cerr << "\n    ouster-cat imu --config config.json:ouster | ouster-to-csv imu";
        std::cerr << "\n";
        std::cerr << "\n    --- display imu data ---";
        std::cerr << "\n    ouster-cat imu --config config.json:ouster | ouster-to-csv imu \\";
        std::cerr << "\n        | csv-shuffle --fields $fields --binary $format --output\\";
        std::cerr << "\n              acceleration/t,acceleration/x,acceleration/y,acceleration/z \\";
        std::cerr << "\n        | csv-plot --binary ul,3f \"-;fields=x,y;color=red\" \\";
        std::cerr << "\n                   \"-;fields=x,,y;color=green\" \"-;fields=x,,,y;color=blue\"";
    }
    else
    {
        std::cerr << "\n";
        std::cerr << "for examples of use try: ouster-to-csv --help --verbose";
    }
    std::cerr << "\n" << std::endl;
}

struct intrinsics_t
{
    ouster::transform_t imu_transform;
    ouster::transform_t lidar_transform;

    intrinsics_t( ouster::OS1::config_t& config )
        : imu_transform( config.imu_intrinsics.imu_to_sensor_transform )
        , lidar_transform( config.lidar_intrinsics.lidar_to_sensor_transform )
    {}
};

template< typename I, typename O >
struct app
{
    static std::string output_fields();
    static std::string output_format() { return comma::csv::format::value< O >(); }
    static std::string output_frame( const intrinsics_t& intrinsics );

    static void process( const I& data_block, comma::csv::binary_output_stream< O >& os );

    static int run( const comma::command_line_options& options )
    {
        if( options.exists( "--output-fields" )) { std::cout << output_fields() << std::endl; return 0; }
        if( options.exists( "--output-format" )) { std::cout << output_format() << std::endl; return 0; }

        std::vector< std::string > config_components = comma::split( options.value< std::string >( "--config", default_config ), ':' );
        std::string config_filename = config_components[0];
        std::string config_path = ( config_components.size() > 1 ? config_components[1] : "" );
        ouster::OS1::config_t config = comma::read_json< ouster::OS1::config_t >( config_filename, config_path, true );

        if( options.exists( "--output-frame" ))
        {
            std::cout << output_frame( intrinsics_t( config )) << std::endl;
            return 0;
        }

        ouster::OS1::init_beam_angle_lut( config.beam_intrinsics );

        output();
        return 0;
    }

    static void output()
    {
        comma::csv::options csv;
        csv.full_xpath = true;
        csv.format( comma::csv::format::value< I >() );
        comma::csv::binary_input_stream< I > is( std::cin, csv );

        comma::csv::options output_csv;
        output_csv.full_xpath = true;
        output_csv.format( comma::csv::format::value< O >() );
        comma::csv::binary_output_stream< O > os( std::cout, output_csv );

        while( is.ready() || ( std::cin.good() && !std::cin.eof() ))
        {
            const I* data_block = is.read();
            if( data_block ) { process( *data_block, os ); }
        }
    }
};

template <>
std::string app< ouster::OS1::azimuth_block_t, ouster::output_lidar_t >::output_fields()
{
    return comma::join( comma::csv::names< ouster::output_lidar_t >( false ), ',' );
}

template<>
std::string app< ouster::OS1::azimuth_block_t, ouster::output_lidar_t >::output_frame( const intrinsics_t& intrinsics )
{
    return comma::join( intrinsics.lidar_transform.frame(), ',' );
}

template<>
void app< ouster::OS1::azimuth_block_t, ouster::output_lidar_t >::process( const ouster::OS1::azimuth_block_t& azimuth_block, comma::csv::binary_output_stream< ouster::output_lidar_t >& os )
{
    static comma::uint32 block_id = 0;
    static comma::uint32 last_encoder_count = 0;

    if( azimuth_block.packet_status == ouster::OS1::packet_status_good )
    {
        if( azimuth_block.encoder_count < last_encoder_count ) { block_id++; }
        last_encoder_count = azimuth_block.encoder_count;
        const ouster::output_azimuth_block_t output_azimuth_block( azimuth_block, block_id );
        const double azimuth_encoder_angle( M_PI * 2 * azimuth_block.encoder_count / ouster::OS1::encoder_ticks_per_rev );
        for( comma::uint16 channel = 0; channel < azimuth_block.data_blocks.size(); ++channel )
        {
            os.write( ouster::output_lidar_t( output_azimuth_block
                                            , ouster::output_data_block_t( azimuth_encoder_angle
                                                                         , azimuth_block.data_blocks[channel]
                                                                         , channel )));
        }
        os.flush();
    }
}

template <>
std::string app< ouster::OS1::imu_block_t, ouster::output_imu_t >::output_fields()
{
    return comma::join( comma::csv::names< ouster::output_imu_t >( true ), ',' );
}

template<>
std::string app< ouster::OS1::imu_block_t, ouster::output_imu_t >::output_frame( const intrinsics_t& intrinsics )
{
    return comma::join( intrinsics.imu_transform.frame(), ',' );
}

template<>
void app< ouster::OS1::imu_block_t, ouster::output_imu_t >::process( const ouster::OS1::imu_block_t& data_block, comma::csv::binary_output_stream< ouster::output_imu_t >& os )
{
    os.write( ouster::output_imu_t( data_block ));
    os.flush();
}

int main( int ac, char** av )
{
    try
    {
        comma::command_line_options options( ac, av, usage );
        if( options.exists( "--bash-completion" ) ) bash_completion( ac, av );

        std::vector< std::string > unnamed = options.unnamed( "--help,-h,--output-fields,--output-format,--output-frame,--verbose,-v", "-.*" );
        if( unnamed.size() == 1 )
        {
            if( unnamed[0] == "lidar" ) { return app< ouster::OS1::azimuth_block_t, ouster::output_lidar_t >::run( options ); }
            else if( unnamed[0] == "imu" ) { return app< ouster::OS1::imu_block_t, ouster::output_imu_t >::run( options ); }
        }
        std::cerr << "ouster-to-csv: require one of lidar or imu" << std::endl;
        return 1;
    }
    catch( std::exception& ex ) { std::cerr << "ouster-to-csv: " << ex.what() << std::endl; }
    catch( ... ) { std::cerr << "ouster-to-csv: unknown exception" << std::endl; }
    return 1;
}

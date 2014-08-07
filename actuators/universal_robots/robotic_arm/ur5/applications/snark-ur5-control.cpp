// This file is part of snark, a generic and flexible library for robotics research
// Copyright (c) 2011 The University of Sydney
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. All advertising materials mentioning features or use of this software
//    must display the following acknowledgement:
//    This product includes software developed by the The University of Sydney.
// 4. Neither the name of the The University of Sydney nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
// HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
// IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/optional.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
#include <comma/application/command_line_options.h>
#include <comma/csv/stream.h>
#include <comma/base/types.h>
#include <comma/math/compare.h>
#include <comma/visiting/apply.h>
#include <comma/name_value/ptree.h>
#include <comma/name_value/parser.h>
#include <comma/io/stream.h>
#include <comma/io/publisher.h>
#include <comma/csv/stream.h>
#include <comma/string/string.h>
#include <comma/application/signal_flag.h>
#include "../../traits.h"
#include "../../commands.h"
#include "../../commands_handler.h"
#include "../../inputs.h"
#include "../../units.h"
extern "C" {
    #include "../../simulink/Arm_Controller.h"
}
#include "../../simulink/traits.h"

/* External inputs (root inport signals with auto storage) */
extern ExtU_Arm_Controller_T Arm_Controller_U;

/* External outputs (root outports fed by signals with auto storage) */
extern ExtY_Arm_Controller_T Arm_Controller_Y;

static const char* name() {
    return "robot-arm-daemon: ";
}

namespace impl_ {

template < typename T >
std::string str(T t) { return boost::lexical_cast< std::string > ( t ); }
    
} // namespace impl_ {


namespace arm = snark::ur::robotic_arm;
using arm::handlers::input_primitive;
using arm::handlers::result;

void usage(int code=1)
{
    std::cerr << std::endl;
    std::cerr << name() << std::endl;
    std::cerr << "example: socat tcp-listen:9999,reuseaddr EXEC:\"snark-ur10-control --id 7 -ip 192.168.0.10 -p 8888\" " << name() << " " << std::endl;
    std::cerr << "          Listens for commands from TCP port 9999, process command and send control string to 192.168.0.10:8888" << std::endl;
    std::cerr << "options:" << std::endl;
    std::cerr << "    --help,-h:            show this message" << std::endl;
    std::cerr << "    --versbose,-v:        show messages to the robot arm - angles are changed to degrees." << std::endl;
    std::cerr << "*   --id=:                ID to identify commands, eg. ><ID>,999,set_pos,home;" << std::endl;
    std::cerr << "*   --status-port=|-sp=:  TCP service port the statuses will be broadcasted on. See below." << std::endl;
    std::cerr << "*   --robot-arm-host=:    Host name or IP of the robot arm." << std::endl;
    std::cerr << "*   --robot-arm-port=:    TCP Port number of the robot arm." << std::endl;
    std::cerr << "*   --feedback-host=:     Host name or IP of the robot arm's feedback." << std::endl;
    std::cerr << "*   --feedback-port=:     TCP Port number of the robot arm's feedback." << std::endl;
    std::cerr << "    --sleep=:             Loop sleep value in seconds, default is 0.2s if not specified." << std::endl;
    std::cerr << "*   --config=:            Config file for robot arm, see --output-config." << std::endl;
    std::cerr << "    --output-config=:     Print config format in json." << std::endl;
    // std::cerr << "    --init-force-limit,-ifl:" << std::endl;
    // std::cerr << "                          Force (Newtons) limit when auto initializing, if exceeded then stop auto init." << std::endl;
    typedef arm::current_positions current_positions_t;
    comma::csv::binary< current_positions_t > binary;
    std::cerr << "UR10's status:" << std::endl;
    std::cerr << "   format: " << binary.format().string() << " total size is " << binary.format().size() << " bytes" << std::endl;
    std::vector< std::string > names = comma::csv::names< current_positions_t >();
    std::cerr << "   fields: " << comma::join( names, ','  ) << " number of fields: " << names.size() << std::endl;
    std::cerr << std::endl;
    exit ( code );
}


template < typename T > 
comma::csv::ascii< T >& ascii( )
{
    static comma::csv::ascii< T > ascii_;
    return ascii_;
}


typedef boost::units::quantity< boost::units::si::angular_acceleration > angular_acceleration_t;
typedef boost::units::quantity< boost::units::si::angular_velocity > angular_velocity_t;


class arm_output
{
public:
    typedef arm::current_positions current_positions_t;
private:
    angular_acceleration_t acceleration;
    angular_velocity_t velocity;
    ExtY_Arm_Controller_T& joints;
    current_positions_t& current_positions;
public:
    arm_output( const angular_acceleration_t& ac, const angular_velocity_t& vel,
                ExtY_Arm_Controller_T& output ) : 
                acceleration( ac ), velocity( vel ), joints( output ), 
                current_positions( static_cast< current_positions_t& >( output ) ) 
                {
                    Arm_Controller_initialize();
                }
   ~arm_output() 
    { 
        Arm_Controller_terminate(); 
    }
                
   std::string debug_in_degrees() const
   {
       std::ostringstream ss;
       ss << "debug: movej([";
       for(std::size_t i=0; i<6u; ++i) 
       {
          ss << static_cast< arm::plane_angle_degrees_t >( joints.joint_angle_vector[i] * arm::radian ).value();
          if( i < 5 ) { ss << ','; }
       }
       ss << "],a=" << acceleration.value() << ','
          << "v=" << velocity.value() << ')';
          
          
       return ss.str();
   }
   std::string serialise() const
   {
       static std::string tmp;
       std::ostringstream ss;
       ss << "movej([" << ascii< ExtY_Arm_Controller_T >( ).put( joints, tmp )
          << "],a=" << acceleration.value() << ','
          << "v=" << velocity.value() << ')';
       return ss.str();
   }
   
   void write_arm_status( comma::io::publisher& publisher )
   {
       // write a byte indicating the status, and joint positions
       static comma::csv::binary< current_positions_t > binary("","",true, current_positions );
       static std::vector<char> line( binary.format().size() );
       binary.put( current_positions, line.data() );
       publisher.write( line.data(), line.size());
   }
                
};


void output( const std::string& msg, std::ostream& os=std::cout )
{
    os << msg << std::endl;
}

static arm::status_t arm_status; 
/// Stream to command robot arm
namespace ip = boost::asio::ip;
typedef arm::handlers::commands_handler commands_handler_t;
typedef boost::shared_ptr< commands_handler_t > commands_handler_shared;
static commands_handler_shared commands_handler;

template < typename C >
std::string handle( const std::vector< std::string >& line, std::ostream& os )
{
    C c;
    try
    {
        c = C::ascii().get( line );
    }
    catch( boost::bad_lexical_cast& le ) {
        std::ostringstream ss;
        ss << '<' << comma::join( line, ',' ) << ',' << impl_::str( arm::errors::format_error )
           << ",\"command format error, wrong field type/s, fields: " << c.names() << " - types: "  << c.serialise() << "\";";
        return ss.str();
    }
    catch( comma::exception& ce ) {
        std::ostringstream ss;
        ss << '<' << comma::join( line, ',' ) << ',' << impl_::str( arm::errors::format_error )
           << ",\"command format error, wrong field/s or field type/s, fields: " << c.names() << " - types: "  << c.serialise() << "\";";
        return ss.str();
    }
    catch( ... ) { COMMA_THROW( comma::exception, "unknown error is parsing: " + comma::join( line , ',' ) ); }
       
    comma::dispatch::handler& h_ref( *commands_handler );
    comma::dispatch::dispatched_base& ref( c );
    ref.dispatch_to( h_ref );
    // perform action
    // result ret = arm::action< C >::run( c, os );
    std::ostringstream ss;
    ss << '<' << c.serialise() << ',' << commands_handler->ret.get_message() << ';';
    return ss.str();
}

void process_command( const std::vector< std::string >& v, std::ostream& os )
{
    if( boost::iequals( v[2], "move_cam" ) )         { output( handle< arm::move_cam >( v, os ) ); }
    else if( boost::iequals( v[2], "set_pos" ) )     { output( handle< arm::set_position >( v, os ) ); }
    else if( boost::iequals( v[2], "set_home" ) )    { output( handle< arm::set_home >( v, os ) ); }
    else if( boost::iequals( v[2], "power" ) )       { output( handle< arm::power >( v, os )); }  
    else if( boost::iequals( v[2], "brakes" ) || 
             boost::iequals( v[2], "stop" ) )        { output( handle< arm::brakes >( v, os )); }  
    else if( boost::iequals( v[2], "auto_init" ) )  
    { 
        if( v.size() == arm::auto_init_force::fields ) 
                                                     { output( handle< arm::auto_init_force >( v, os )); }
        else                                         { output( handle< arm::auto_init >( v, os )); } 
    }  
    else if( boost::iequals( v[2], "initj" ) )       { output( handle< arm::joint_move >( v, os )); 
    }  
    // else if( boost::iequals( v[2], "movej" ) )      { output( handle< arm::move_joints >( v, os ) ); }
    else { output( comma::join( v, v.size(), ',' ) + ',' + 
        impl_::str( arm::errors::unknown_command ) + ",\"unknown command found: '" + v[2] + "'\"" ); return; }
}

// /// Connect to the TCP server within the allowed timeout
// /// Needed because comma::io::iostream is not available
// bool tcp_connect( const std::string& host, const std::string& port, 
//                   const boost::posix_time::time_duration& timeout, ip::tcp::iostream& io )
// {
//     using boost::asio::ip::tcp;
//     boost::asio::io_service service;
//     tcp::resolver resolver( service );
//     tcp::resolver::query query( host == "localhost" ? "127.0.0.1" : host, port );
//     tcp::resolver::iterator it = resolver.resolve( query );
//     // Connect and find out if successful or not quickly using timeout
//     io.expires_from_now( timeout );
//     io.connect( it->endpoint() );
//     io.expires_at( boost::posix_time::pos_infin );
//     
//     return io.error() == 0;
// } 

bool ready( comma::io::istream& is )
{
    // std::cerr << "avail: " << is->rdbuf()->in_avail() << std::endl;
    return is->rdbuf()->in_avail() > 0;
}

/// Return null if no status
bool read_status( comma::csv::binary_input_stream< arm::status_t >& iss, 
    comma::io::select& select, comma::io::file_descriptor fd )
{
    /// Within 100ms, we are guranteed a new status, there may already be many statuses waiting to be read
    static const boost::posix_time::milliseconds timeout( 0.1 * 1000000u );
    select.wait( timeout );
    if( !select.read().ready( fd ) ) { 
        std::cerr << "no status received within timeout of " << timeout.total_milliseconds() << "ms" << std::endl; 
        COMMA_THROW( comma::exception, "no status received within timeout of " << timeout.total_milliseconds() << "ms" ); 
    }
    arm_status = *( iss.read() );
    while( iss.has_data() )  { arm_status = *( iss.read() ); }

    // // std::cerr << "rdbuf" << std::endl;
    // iss->read( arm_status.data(), arm::fixed_status::size );
    // if( !iss->good() ) {
    //     // std::cerr  << "not good" << std::endl;
    //     return false;
    // }
    // // Keep reading to get latest data
    // while( ready( iss ) )
    // {
    //     iss->read( arm_status.data(), arm::fixed_status::size );
    //     // std::cerr << "ready again read" << std::endl;
    // }

    if( arm_status.length != arm::fixed_status::size ) {
        std::cerr << name() << "status data alignment check failed" << std::endl; 
        COMMA_THROW( comma::exception, "status data alignment check failed" ); 
    }
    return true;
}

static arm::config config;

class stop_on_exit
{
    std::ostream& os_;
public:
    stop_on_exit( std::ostream& oss ) : os_(oss) {}
    ~stop_on_exit() 
    {
        os_ << "stopj([0.1,0.1,0.1,0.1,0.1,0.1])\n";
        os_ << "power off\n"; // power off too
        os_.flush();
    }

};

void load_config( const std::string& filepath )
{
    std::ifstream config_ifs( filepath.c_str() );
    if( !config_ifs.is_open() ) { COMMA_THROW( comma::exception, "failed to open file: " + filepath ); }

    boost::property_tree::ptree t;
    comma::from_ptree( t, true );
    boost::property_tree::read_json( config_ifs, t );
    comma::from_ptree from_ptree( t, true );
    comma::visiting::apply( from_ptree ).to( config );
}

namespace fs = boost::filesystem;

/// Create a home position file if arm is running and is in home position
void home_position_check( const arm::status_t& status, const std::string& homefile )
{
    // static std::vector< arm::plane_angle_t > home_position; /// store home joint positions in radian
    static const fs::path path( homefile );
    static const arm::plane_angle_t epsilon = static_cast< arm::plane_angle_t >( 2.0 * arm::degree );
    
    static std::vector< arm::plane_angle_t > home_position; /// store home joint positions in radian
    if( home_position.empty() )
    {
        home_position.push_back( static_cast< arm::plane_angle_t >( config.continuum.home_position[0] * arm::degree ) );
        home_position.push_back( static_cast< arm::plane_angle_t >( config.continuum.home_position[1] * arm::degree ) );
        home_position.push_back( static_cast< arm::plane_angle_t >( config.continuum.home_position[2] * arm::degree ) );
        home_position.push_back( static_cast< arm::plane_angle_t >( config.continuum.home_position[3] * arm::degree ) );
        home_position.push_back( static_cast< arm::plane_angle_t >( config.continuum.home_position[4] * arm::degree ) );
        home_position.push_back( static_cast< arm::plane_angle_t >( config.continuum.home_position[5] * arm::degree ) );
    }


    if( status.is_running() )
    {
        // TODO, either create or delete the home position file
        bool is_home = true;
        for( std::size_t i=0; i<=arm::joints_num; ++i ) {
            if( !comma::math::equal( status.joint_angles[i], home_position[i], epsilon ) ){ is_home=false; break; }
        }

        if( is_home ){ std::ofstream( homefile.c_str(), std::ios::out | std::ios::trunc ); } // create
        else { fs::remove( path ); } // remove
    }
}

int main( int ac, char** av )
{
    
    comma::signal_flag signaled;
    
    comma::command_line_options options( ac, av );
    if( options.exists( "-h,--help" ) ) { usage( 0 ); }

    if( options.exists( "--output-config") )
    {
        boost::property_tree::ptree t;
        comma::to_ptree to_ptree( t );
        comma::visiting::apply( to_ptree ).to( config );
        boost::property_tree::write_json( std::cout, t );  
    }
    
    using boost::posix_time::microsec_clock;
    using boost::posix_time::seconds;
    using boost::posix_time::ptime;
    using boost::asio::ip::tcp;

    double acc = 0.5;
    double vel = 0.1;

    std::cerr << name() << "started" << std::endl;
    try
    {
        arm_output output( acc * angular_acceleration_t::unit_type(), vel * angular_velocity_t::unit_type(),
                       Arm_Controller_Y );
    
        comma::uint16 rover_id = options.value< comma::uint16 >( "--id" );
        double sleep = 0.1; // seconds
        if( options.exists( "--sleep" ) ) { sleep = options.value< double >( "--sleep" ); };

        comma::uint32 listen_port = options.value< comma::uint32 >( "--status-port,-sp" );
        
        bool verbose = options.exists( "--verbose,-v" );

        std::string config_file = options.value< std::string >( "--config" );
        load_config( config_file );
        
        /// home position file
        const arm::continuum_t& continuum = config.continuum;
        if( continuum.work_directory.empty() ) { std::cerr << name() << "cannot find home position directory! exiting!" <<std::endl; return 1; }
        fs::path dir( continuum.work_directory );
        if( !fs::exists( dir ) || !fs::is_directory( dir ) ) { std::cerr << name() << "work_directory must exists: " << continuum.work_directory << std::endl; return 1; }

        for( std::size_t j=0; j<arm::joints_num; ++j )
        {
            std::cerr << name() << "home joint " << j << " - " << continuum.home_position[j] << '"' << std::endl;
        }

        std::string arm_conn_host = options.value< std::string >( "--robot-arm-host" );
        std::string arm_conn_port = options.value< std::string >( "--robot-arm-port" );
        std::string arm_feedback_host = options.value< std::string >( "--feedback-host" );
        std::string arm_feedback_port = options.value< std::string >( "--feedback-port" );
        
        boost::scoped_ptr< comma::io::ostream > poss;
        try
        {
            const std::string cmd_str = "tcp:" + arm_conn_host + ':' + arm_conn_port;
            std::cerr << name() << "connecting to the robotic arm command channel: " << cmd_str << std::endl;
            poss.reset( new comma::io::ostream( cmd_str, comma::io::mode::ascii, comma::io::mode::non_blocking ) );
        }
        catch( comma::exception& e )
        {
            std::cerr << name() << "failed to connect to tcp:" << arm_conn_host << ':' << arm_conn_port << std::endl;
            return 1;
        }
        comma::io::ostream& robot_arm = *poss;

        stop_on_exit on_exit( *robot_arm );

        // create tcp server for broadcasting status
        std::ostringstream ss;
        ss << "tcp:" << listen_port;
        comma::io::publisher publisher( ss.str(), comma::io::mode::binary );

        arm::inputs inputs( rover_id );

        typedef std::vector< std::string > command_vector;
        const comma::uint32 usec( sleep * 1000000u );
        
        std::string status_conn = "tcp:" + arm_feedback_host + ':' + arm_feedback_port;
        std::cerr << name() << "status connection to feedback status: " << status_conn << std::endl;
        comma::io::istream status_stream( status_conn, comma::io::mode::binary );
        comma::csv::options csv_in;
        csv_in.full_xpath = true;
        csv_in.format( comma::csv::format::value< arm::status_t >( "", true ) );
        comma::csv::binary_input_stream< arm::status_t > istream( *status_stream, csv_in );
        comma::io::select select;
        select.read().add( status_stream.fd() );

        arm::handlers::auto_initialization auto_init( arm_status, *robot_arm, istream, select, status_stream.fd(), signaled, inputs, continuum.work_directory );
        auto_init.set_app_name( name() );
        std::cerr << "fd start: " << status_stream.fd() << std::endl;
        if( options.exists( "--init-force-limit,-ifl" ) ){ auto_init.set_force_limit( options.value< double >( "--init-force-limit,-ifl" ) ); }
        commands_handler.reset( new commands_handler_t( Arm_Controller_U, arm_status, *robot_arm, auto_init ) );

        while( !signaled && std::cin.good() )
        {
            if( !status_stream->good() ) { 
                std::cerr << name() << "status connection to robot-arm failed" << std::endl;
                COMMA_THROW( comma::exception, "status connection to robot arm failed." ); 
            }

            if( read_status( istream, select, status_stream.fd() ) )
            { 
                // std::cerr << name() << "robotmode: " << arm_status.mode_str() << std::endl;
                // boost::property_tree::ptree t;
                // comma::to_ptree to_ptree( t );
                // comma::visiting::apply( to_ptree ).to( arm_status );
                // boost::property_tree::write_json( std::cerr, t, false );    
                home_position_check( arm_status, auto_init.home_filepath() );
            }
            
            try { inputs.read(); }
            catch(...) { COMMA_THROW( comma::exception, "reading from stdcin failed." ); }
            // Process commands into inputs into the system
            if( !inputs.is_empty() )
            {
                const command_vector v = inputs.front();
                inputs.pop();
                //std::cerr << name() << " got " << comma::join( v, ',' ) << std::endl;
                process_command( v, *robot_arm );

            }
            // Run simulink code
            Arm_Controller_step();
            
            // We we need to send command to arm
            if( Arm_Controller_Y.command_flag > 0 )
            {
                if( verbose ) { std::cerr << name() << output.debug_in_degrees() << std::endl; }
                *robot_arm << output.serialise() << std::endl;
                robot_arm->flush();
                Arm_Controller_U.motion_primitive = real_T( input_primitive::no_action );
            }
            else if( Arm_Controller_Y.command_flag < 0 ) {
                std::cerr << name() << "command cannot execute as it will cause a collision!" << std::endl;
            }
            
            // reset inputs
            memset( &Arm_Controller_U, 0, sizeof( ExtU_Arm_Controller_T ) );
            // send out arm's current status: code and joint positions
            output.write_arm_status( publisher );

            if( sleep > 0 ) usleep( usec );
        }

        std::cerr << name() << "exiting" << std::endl;
        *robot_arm << "power off\n";
        robot_arm->flush();
        publisher.close();
    }
    catch( comma::exception& ce ) { std::cerr << name() << ": exception thrown: " << ce.what() << std::endl; return 1; }
    catch( std::exception& e ) { std::cerr << name() << ": unknown exception caught: " << e.what() << std::endl; return 1; }
    catch(...) { std::cerr << name() << ": unknown exception." << std::endl; return 1; }
    
}

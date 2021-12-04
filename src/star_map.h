#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdio.h>

#include <watdefs.h>
#include <jpleph.h>
#include <calceph.h> // Using 3.5

#include <date.h>
#include <date/julian.h>
#include <date/tz.h>

#include <pangolin/var/var.h>
#include <pangolin/var/varextra.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/display/default_font.h>
#include <pangolin/handler/handler.h>

#include <sophus/so3.hpp>

constexpr double AU_IN_KM = 1.49597870691e+8;
constexpr double J2000_jd = 2451545.0;

// Frames of reference:
//
// J2000:
//        International rotational coordinate system defined by orientation of earth at
//        (around) noon January 1st 2000 at the greenich meridian.
//
//        This inertial (non-rotating) reference frame is defined by the following orthogonal
//        axis directions:
//          z-axis: normal to the the mean equatorial plane, or equivelently the mean rotational axis.
//                  Note that the mean is used because the Earth wiggles slightly on this axis
//                  (nutations, processions, polar motion) and adjustments for times beyond J2000 are
//                  simpler from the mean and not true orientation at this time.
//          x-axis: Intersection of mean equatorial and ecliptic planes (the one of two closest to the sun,
//                  which is named the vernal equinox). The ecliptic plane is the plane around the sun that
//                  the Earth orbits within. For J2000 at noon on the Greenwich meridian, the sun is at it's
//                  highest in the sky and the earth-sun vector lies within the plane of the Greenwich
//                  meridian. Therefore the intersection of the Greenwich meridian and the mean equatorial
//                  plane also coincide with this direction. i.e the x-axis points at 0 latitude, 0 longtidude
//                  at the time defined by J2000.
//          y-axis: J2000 is a right-handed coordinate system and the y-axis can be found through the cross
//                  product of the z-axis with the x-axis.
//
//        see: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/04_concepts.pdf
//
// Earth(jd):
//        We will define a reference frame `Earth` at time jd (in JED units of time, often called Julian Day
//        or JD) as the frame which holds the Earth static with respect to the J2000 frame over time.
//        i.e. Earth(J2000_jd)     = J2000.
//             Earth(J2000_jd+1.0) = Rz(2*M_PI) * J2000 [+ small pertubations]
//             Earth(jd)  = Rz(2*M_PI*(jd-J2000_jd)) * J2000 [+ small pertubations]
//
//             where:
//                J2000_jd is the time of J2000 in JED units
//
//                Rz(x) is a rotation matrix defined by a rotation x around the z-axis obeying the
//                right-hand rule.
//
// LatLon2000(lat,lon):
//        The LatLon reference frame is a convenience right-handed frame for relating points on the surface
//        of the earth at the time defined by J2000. We will define it as a normalizing frame such that:
//          z-axis: is oriented towards the lan-long vector,
//          y-axis: is oriented in the plane containing the north pole and lan-long vector,
//                  orthogonal to the z-axis
//          x-axis: can be found through the cross product of the y-axis with the z-axis.
//
// LatLon(jd, lat, lon):
//        This is the evolution of the LatLon2000(lat,lon) frame with time, such that the frame defined by
//        the LatLon2000 convention holds for time t (in JED units of time).
//
//        LatLon(t, lat, lon) = R_Earth(t)_J2000 * LatLon2000(lat,lon),
//
//        where R_Earth(t)_J2000 is the rotation matrix that transforms the J2000 frame to the Earth(t) frame.

// Return rotation transform which takes a (lat,lon) in frame J2000 into frame LatLon2000 as defined above
template<typename T>
Sophus::SO3<T> R_LatLon2000_J2000(T lat, T lon)
{
    return Sophus::SO3<T>::rotY(lat) * Sophus::SO3<T>::rotZ(lon);
}

// Return rotation transform which takes a (lat,lon) in frame LatLon2000 into frame J2000 as defined above
template<typename T>
Sophus::SO3<T> R_J2000_LatLon2000(T lat, T lon)
{
    return R_LatLon2000_J2000<T>(lat,lon).inverse();
}

// Return rotation transform which takes frame J2000 into Earth(jd) as defined above
template<typename T>
Sophus::SO3<T> R_Earth_J2000(T jd)
{
    return Sophus::SO3<T>::rotZ(jd * Sophus::Constants<double>::tau());
}

// Return point on (lat,lon) vector with magnitude alt_km in J2000 reference
template<typename T>
Eigen::Vector3<T> LatLon_J2000(T lat, T lon, T alt_km = 6371.009)
{
    return R_J2000_LatLon2000<T>(lat, lon) * Eigen::Vector3<T>(0.0, 0.0, alt_km);
}

// Return point on (lat,lon) vector with magnitude alt_km in Earth(jd) reference
template<typename T>
Eigen::Vector3<T> LatLon_Earth(T jd, T lat, T lon, T alt_km = 6371.009)
{
    return R_Earth_J2000<T>(jd) * LatLon_J2000<T>(lat, lon, alt_km);
}

void
test_star_map2()
{
    auto zt = date::make_zoned(date::current_zone(), std::chrono::system_clock::now());
    auto ld = date::floor<date::days>(zt.get_local_time());
    julian::year_month_day ymd{ld};
    auto time = date::make_time(zt.get_local_time() - ld);
    std::cout << ymd << ' ' << time << '\n';
}

double utc_timepoint_to_jd(const std::chrono::system_clock::time_point& utc_time = std::chrono::system_clock::now() )
{
    const auto timepoint_utc_day_start = date::floor<date::days>(utc_time);
    const auto ymd = date::year_month_day{timepoint_utc_day_start};

    const auto timepoint_utc_noon = timepoint_utc_day_start + std::chrono::hours(12);
    const long seconds_offset = std::chrono::duration_cast<std::chrono::seconds>(utc_time-timepoint_utc_noon).count();
    const double day_fraction_from_noon = seconds_offset / (60*60*24.0);

    const long JD_noon = dmy_to_day(uint(ymd.day()), uint(ymd.month()), int(ymd.year()), CALENDAR_JULIAN_GREGORIAN);
    const double JD = JD_noon + day_fraction_from_noon;
    return JD;
}



void attempt2()
{

}

template<typename Derived>
void glTranslate(const Eigen::DenseBase<Derived>& x)
{
    glTranslated(x[0], x[1], x[2]);
}

void RenderSolarSystem(t_calcephbin *peph, double jd0, double jd_offset)
{
    struct ObjectOfInterest
    {
        int naif_id;
        double radius_au;
    };

//    std::vector<ObjectOfInterest> objects = {
//        {NAIFID_MERCURY,     4900 / AU_IN_KM / 2.0}, // Mercury center
//        {NAIFID_VENUS,      12100 / AU_IN_KM / 2.0}, // Venus center
//        {NAIFID_EARTH,      12800 / AU_IN_KM / 2.0}, // ...
//        {NAIFID_MARS,        6800 / AU_IN_KM / 2.0},
//        {NAIFID_JUPITER,   143000 / AU_IN_KM / 2.0},
//        {NAIFID_SATURN,    120500 / AU_IN_KM / 2.0},
//        {NAIFID_URANUS,     51100 / AU_IN_KM / 2.0},
//        {NAIFID_NEPTUNE,    49500 / AU_IN_KM / 2.0},
//        {NAIFID_PLUTO,     2376.6 / AU_IN_KM / 2.0}, // Pluto center
//        {NAIFID_SUN,    1391000.0 / AU_IN_KM / 2.0}, // Sun center
//        {NAIFID_MOON,     34748.0 / AU_IN_KM / 2.0}  // Moon center
//    };

    std::vector<ObjectOfInterest> objects = {
        {1,     4900 / AU_IN_KM / 2.0}, // Mercury center
        {2,      12100 / AU_IN_KM / 2.0}, // Venus center
        {3,      12800 / AU_IN_KM / 2.0}, // ...
        {4,        6800 / AU_IN_KM / 2.0},
        {5,   143000 / AU_IN_KM / 2.0},
        {6,    120500 / AU_IN_KM / 2.0},
        {7,     51100 / AU_IN_KM / 2.0},
        {8,    49500 / AU_IN_KM / 2.0},
        {9,     2376.6 / AU_IN_KM / 2.0}, // Pluto center
        {10,    1391000.0 / AU_IN_KM / 2.0}, // Sun center
//        {NAIFID_MOON,     34748.0 / AU_IN_KM / 2.0}  // Moon center
    };

//    const Eigen::Vector<double,11> body_dia_km = {
//        0.0,   // Solar System
//          4900,  // mercury
//         12100, // venus
//         12800, // ...
//          6800,
//        143000,
//        120500,
//         51100,
//         49500,
//        2376.6, // pluto
//        34748.0, // moon
//     1391000.0 // sun
//    };

//    const Eigen::Vector<double,11> body_rad_au =body_dia_km / AU_IN_KM / 2.0;


    for( const auto& obj : objects)
    {
        if(obj.naif_id == NAIFID_SUN) continue;

        Eigen::Vector<double,6> P_sun;
        Eigen::Vector<double,6> Euler_sun;

//        jpl_pleph( p, jd, i, 12, P_sun.data(), 0);
//        std::cout << "-----------" << std::endl;
        calceph_compute_unit(peph, jd0, jd_offset, obj.naif_id, 11, CALCEPH_UNIT_AU | CALCEPH_UNIT_DAY, P_sun.data());
        calceph_orient_order(peph, jd0, jd_offset, NAIFID_EARTH, CALCEPH_USE_NAIFID | CALCEPH_UNIT_SEC | CALCEPH_UNIT_RAD | CALCEPH_OUTPUT_NUTATIONANGLES, 0, Euler_sun.data());

//        std::cout << P_sun.transpose() << std::endl;
//        std::cout << Sun_P.transpose() << std::endl;
//        std::cout << (P_sun+Sun_P).transpose() << std::endl;

        glPushMatrix();
        glTranslate(P_sun.head<3>());

//        const double rad = body_rad_au[i] * 100.0;
        const double rad = obj.radius_au * 1000.0;

        pangolin::glDrawColouredCube(-rad, +rad);
        glPopMatrix();

        // Center
        pangolin::glDrawAxis(1.0);
    }
}

void test_star_map()
{
    const char* files[] = {
//        "/Users/stevenlovegrove/Downloads/eph/de440.bsp",
//        "/Users/stevenlovegrove/Downloads/eph/de441_part-1.bsp",
//        "/Users/stevenlovegrove/Downloads/eph/de441_part-2.bsp",
//        "/Users/stevenlovegrove/Downloads/eph/de-403-masses.tpc",
//        "/Users/stevenlovegrove/Downloads/eph/moon_pa_de430_1550-2650.bpc",
//        "/Users/stevenlovegrove/Downloads/eph/lnxp1600p2200.405",
          "/Users/stevenlovegrove/Downloads/eph/linux_m13000p17000.441",
    };

//    t_calcephbin *peph = calceph_open("/Users/stevenlovegrove/code/telescope/data/de440/linux_p1550p2650.440");
    t_calcephbin *peph = calceph_open_array(std::size(files), files);
    if (!peph) throw std::runtime_error("Couldn't open ephemeris file.");

    if(!calceph_prefetch(peph))
        throw std::runtime_error("Unable to prefetch ephemeris contents");

    double start_jd, end_jd;
    int countinuous;
    if (!calceph_gettimespan(peph,  &start_jd, &end_jd, &countinuous))
        throw std::runtime_error("Couldn't determine ephemeris time range.");

//    std::cout << std::fixed << start_jd << " - " << end_jd << std::endl;
//    exit(0);

//    const unsigned JPL_MAX_N_CONSTANTS = 1018;
//    char nams[JPL_MAX_N_CONSTANTS][6], buff[102];
//    double vals[JPL_MAX_N_CONSTANTS];

//    void* ephem = jpl_init_ephemeris( "/Users/stevenlovegrove/code/telescope/data/de440/linux_p1550p2650.440", nams, vals);
//    const double start_jd = jpl_get_double( ephem, JPL_EPHEM_START_JD);
//    const double end_jd = jpl_get_double( ephem, JPL_EPHEM_END_JD);

    const double now_jd = utc_timepoint_to_jd();
    const double now_plus_1y_jd = now_jd - 365.0;


    pangolin::CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(-0,0.5,-3, 0,0,0, pangolin::AxisY)
    );

    const int UI_WIDTH = 20* pangolin::default_font().MaxWidth();

    pangolin::View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, 640.0f/480.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<double> ui_jd("ui.JD", 0.0, 0.0, 360.0);

    while( !pangolin::ShouldQuit() )
    {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if(d_cam.IsShown()) {
          d_cam.Activate(s_cam);
          RenderSolarSystem(peph, now_jd, ui_jd);
      }

      pangolin::FinishFrame();
    }

//    jpl_close_ephemeris( ephem);
    calceph_close(peph);

}

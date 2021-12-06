#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdio.h>
#include <type_traits>

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

#include <sophus/se3.hpp>

constexpr double AU_IN_KM = 1.49597870691e+8;
constexpr double J2000_jd = 2451545.0;
constexpr double EARTH_MEAN_RADIUS_KM = 6371.009;

// Frames of reference:
//
// J2000:
//        International rotational coordinate system defined by orientation of earth at
//        (around) noon January 1st 2000 at the greenich meridian.
//
//        This inertial (non-rotating) reference frame is defined by the following orthogonal
//        axis directions:
//          z-axis: normal to the the mean equatorial plane, or equivelently the mean rotational axis.
//                  +ive direction faces north and is consistent with the right hand rule wrt body
//                  rotation around z.
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
//
//
// LunaME:
//        Mean Earth / Polar Axis (ME) reference coordinate system.
//        https://lunar.gsfc.nasa.gov/library/LunCoordWhitePaper-10-08.pdf
//
//        z-axis: normal to the the mean equatorial plane, or equivelently the mean rotational axis.
//                +ive direction faces north and is consistent with the right hand rule wrt body
//                rotation around z.
//        x-axis: Intersection of equitorial plane and Lunar Prime Meridian, where the Prime Meridian
//                is the plane of mean earth direction.
//        y-axis: Ad a right handed system, can be found from the cross product of z and x axes.
//
// LunaPA:
//        ... not sure exactly but this is how the orientation of moon is defined for JPL ephemeris


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

// Return point on (lat,lon) vector with magnitude alt_km in J2000 reference
template<typename T>
Eigen::Vector3<T> LatLon_J2000(T lat, T lon, T radius_km = EARTH_MEAN_RADIUS_KM)
{
    return R_J2000_LatLon2000<T>(lat, lon) * Eigen::Vector3<T>(0.0, 0.0, radius_km);
}

template<typename P>
struct TimeJED
{
    P JED() const {
        return jed;
    }

    P JD_Since_JD2000() const {
        return jed - static_cast<P>(J2000_jd);
    }

    P jed;
};

// Return rotation transform which takes frame J2000 into Earth(jd) as defined above
template<typename P, typename Time>
Sophus::SO3<P> R_Earth_J2000(const Time t)
{
    return Sophus::SO3<P>::rotZ(t.JD_Since_JD2000() * Sophus::Constants<double>::tau());
}

template<typename P>
Sophus::SO3<P> R_J2000_LunaPA(const Eigen::Vector3<P>& libration)
{
    // https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf eqn 8 for interpreting libration angles

    const double phi = libration[0];
    const double theta = libration[1];
    const double psi = libration[2];
    return Sophus::SO3d::rotZ(phi) * Sophus::SO3d::rotX(theta) * Sophus::SO3d::rotZ(psi);
}

// Return point on (lat,lon) vector with magnitude alt_km in Earth(jd) reference
template<typename P, typename Time>
Eigen::Vector3<P> LatLon_Earth(const Time t, P lat, P lon, P radius_km = EARTH_MEAN_RADIUS_KM)
{
    return R_Earth_J2000<P,Time>(t) * LatLon_J2000<P>(lat, lon, radius_km);
}

void test_star_map2()
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

template<typename Derived>
void glTranslate(const Eigen::DenseBase<Derived>& x)
{
    glTranslated(x[0], x[1], x[2]);
}

template<typename P>
void glLoadMatrix(const Eigen::Matrix<P,4,4,Eigen::ColMajor>& T_parent_child)
{
    if constexpr( std::is_same_v<P,double>) {
        glLoadMatrixd(T_parent_child.data());
    } else if constexpr( std::is_same_v<P,float>) {
        glLoadMatrixf(T_parent_child.data());
    }
}

template<typename P>
void glLoadMatrix(const Sophus::SE3<P>& T_parent_child)
{
    glLoadMatrix(T_parent_child.matrix());
}

template<typename P>
void glMultMatrix(const Eigen::Matrix<P,4,4,Eigen::ColMajor>& T_parent_child)
{
    if constexpr( std::is_same_v<P,double>) {
        glMultMatrixd(T_parent_child.data());
    } else if constexpr( std::is_same_v<P,float>) {
        glMultMatrixf(T_parent_child.data());
    }
}

template<typename P>
void glMultMatrix(const Sophus::SE3<P>& T_parent_child)
{
    glMultMatrix(T_parent_child.matrix());
}

Eigen::Vector3d BodyPosition(t_calcephbin *peph, double jd0, double jd_offset, int body_naif, int center_naif)
{
    Eigen::Vector<double,6> Body_center;
    calceph_compute_unit(peph, jd0, jd_offset, body_naif, center_naif, CALCEPH_USE_NAIFID | CALCEPH_UNIT_AU | CALCEPH_UNIT_DAY, Body_center.data());
    return Body_center.head<3>();
}

void RenderSolarSystem(t_calcephbin *peph, double jd0, double jd_offset, int center_body = NAIFID_SUN, double body_scale = 1.0)
{
    struct ObjectOfInterest
    {
        int naif_id;
        double radius_au;
    };

    const static std::vector<ObjectOfInterest> objects = {
        {NAIFID_MERCURY_BARYCENTER,     4900 / AU_IN_KM / 2.0},
        {NAIFID_VENUS_BARYCENTER,      12100 / AU_IN_KM / 2.0},
        {NAIFID_EARTH,                 12800 / AU_IN_KM / 2.0},
        {NAIFID_MARS_BARYCENTER,        6800 / AU_IN_KM / 2.0},
        {NAIFID_JUPITER_BARYCENTER,   143000 / AU_IN_KM / 2.0},
        {NAIFID_SATURN_BARYCENTER,    120500 / AU_IN_KM / 2.0},
        {NAIFID_URANUS_BARYCENTER,     51100 / AU_IN_KM / 2.0},
        {NAIFID_NEPTUNE_BARYCENTER,    49500 / AU_IN_KM / 2.0},
        {NAIFID_PLUTO_BARYCENTER,     2376.6 / AU_IN_KM / 2.0},
        {NAIFID_SUN,               1391000.0 / AU_IN_KM / 2.0},
        {NAIFID_MOON,                 3474.8 / AU_IN_KM / 2.0},
//        {4660,               100*0.165000432 / AU_IN_KM / 2.0}
    };

    for( const auto& obj : objects)
    {
        if(obj.naif_id == NAIFID_SUN) continue;

        const Eigen::Vector<double,3> P_center = BodyPosition(peph, jd0, jd_offset, obj.naif_id, center_body);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        Sophus::SE3d T_wo;
        T_wo.translation() = P_center.head<3>();

        if(obj.naif_id == NAIFID_EARTH) {
//            // TODO: add nutation angle adjustment
//            Eigen::Vector<double,6> nutation;
//            calceph_orient_unit(peph, jd0, jd_offset, NAIFID_EARTH, CALCEPH_USE_NAIFID | CALCEPH_OUTPUT_NUTATIONANGLES | CALCEPH_UNIT_RAD | CALCEPH_UNIT_SEC, nutation.data() );

            T_wo.so3() = R_Earth_J2000<double>(TimeJED<double>{jd0+jd_offset}).inverse().matrix();
        }else if(obj.naif_id == NAIFID_MOON) {
            Eigen::Vector<double,6> libration;
            calceph_orient_unit(peph, jd0, jd_offset, NAIFID_MOON, CALCEPH_USE_NAIFID | CALCEPH_OUTPUT_EULERANGLES | CALCEPH_UNIT_RAD | CALCEPH_UNIT_SEC, libration.data() );
            T_wo.so3() = R_J2000_LunaPA<double>(libration.head<3>());
        }
        glMultMatrix(T_wo);

        const double rad = obj.radius_au * body_scale;

        pangolin::glDrawColouredCube(-rad, +rad);
        glPopMatrix();

        // Center
        pangolin::glDrawAxis(1.0);
    }
}

// https://svs.gsfc.nasa.gov/4851 good resource for celestial textures

void test_star_map()
{
    const char* files[] = {
//        "/Users/stevenlovegrove/Downloads/eph/de440.bsp",
//        "/Users/stevenlovegrove/Downloads/eph/de441_part-1.bsp",
//        "/Users/stevenlovegrove/Downloads/eph/de441_part-2.bsp",
//        "/Users/stevenlovegrove/Downloads/eph/de-403-masses.tpc",
//        "/Users/stevenlovegrove/Downloads/eph/moon_pa_de430_1550-2650.bpc",
//        "/Users/stevenlovegrove/Downloads/eph/lnxp1600p2200.405",
        "/Users/stevenlovegrove/code/telescope/data/ephemeris/linux_p1550p2650.440",
//        "/Users/stevenlovegrove/code/telescope/data/ephemeris/2004660.bsp", // 4660 Nereus
//          "/Users/stevenlovegrove/Downloads/eph/linux_m13000p17000.441",
//        "/Users/stevenlovegrove/Downloads/horizons_4660.txt", // doesn't seem to work
    };



    t_calcephbin *peph = calceph_open_array(std::size(files), files);
    if (!peph) throw std::runtime_error("Couldn't open ephemeris file.");

    if(!calceph_prefetch(peph))
        throw std::runtime_error("Unable to prefetch ephemeris contents");

    double start_jd, end_jd;
    int countinuous;
    if (!calceph_gettimespan(peph,  &start_jd, &end_jd, &countinuous))
        throw std::runtime_error("Couldn't determine ephemeris time range.");

    const double now_jd = utc_timepoint_to_jd();
    const double now_plus_1y_jd = now_jd - 365.0;

    pangolin::CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.01,100),
      pangolin::ModelViewLookAt(-0,0.5,-3, 0,0,0, pangolin::AxisY)
    );

    const int UI_WIDTH = 20* pangolin::default_font().MaxWidth();

    pangolin::View& d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, 640.0f/480.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<double> ui_jd("ui.JD", 0.0, 0.0, 360.0);
    pangolin::Var<double> ui_body_scale("ui.body_scale", 1000.0, 1.0, 1000.0);
    pangolin::Var<int> ui_center_body("ui.center_body", NAIFID_SUN, NAIFID_MERCURY_BARYCENTER, NAIFID_SUN);

    while( !pangolin::ShouldQuit() )
    {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if(d_cam.IsShown()) {
          d_cam.Activate(s_cam);
          RenderSolarSystem(peph, now_jd, ui_jd, ui_center_body, ui_body_scale);
      }

      pangolin::FinishFrame();
    }

    calceph_close(peph);

}

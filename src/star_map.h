#include <iostream>
#include <chrono>
#include <iomanip>
#include <stdio.h>

#include <watdefs.h>
#include <jpleph.h>
#include <calceph.h>

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

#define AU_IN_KM 1.49597870691e+8

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
    double AU;
    t_calcephbin *peph;

    peph = calceph_open("/Users/stevenlovegrove/code/telescope/data/de440/linux_p1550p2650.440");
    if (peph)
    {
        if (calceph_getconstant(peph, "AU", &AU))
        {
            printf("AU=%23.16E\n", AU);
        }

        calceph_close(peph);
    }
}

template<typename Derived>
void glTranslate(const Eigen::DenseBase<Derived>& x)
{
    glTranslated(x[0], x[1], x[2]);
}

void RenderSolarSystem(void* p, double jd)
{
    attempt2();
    return;

    const double au_in_km = 149598073.0;

    const Eigen::Vector<double,12> body_dia_km = {
        0.0,
        4900,  // mercury
        12100, // venus
        12800, // ...
        6800,
        143000,
        120500,
        51100,
        49500,
        2376.6, // pluto
        34748.0, // moon
        1391000.0 // sun
    };

    const Eigen::Vector<double,12> body_rad_au =body_dia_km / au_in_km / 2.0;

    for( size_t i = 1; i < 12; i++)
    {
        Eigen::Vector<double,6> P_sun;
        Eigen::Vector<double,6> sun_P;

        jpl_pleph( p, jd, i, 12, P_sun.data(), 0);
//        jpl_pleph( p, jd, 12, i, sun_P.data(), 0);

//        std::cout << "--------------- " << i << std::endl;
//        std::cout << P_sun.transpose() << std::endl;
//        std::cout << sun_P.transpose() << std::endl;

        glPushMatrix();
        glTranslate(P_sun.head<3>());

        const double rad = body_rad_au[i] * 100.0;
        if(i==11) {
            pangolin::glDrawAxis(rad);
        }else{
            pangolin::glDrawColouredCube(-rad, +rad);
        }
        glPopMatrix();

        //        JPL_EPHEM_EARTH_MOON_RATIO

//        static const char *object_names[] = {
//            "SSBar", "Mercu", "Venus", "EMB  ", "Mars ",
//            "Jupit", "Satur", "Uranu", "Neptu", "Pluto",
//            "Moon " };
    }
//    exit(0);
}

void test_star_map()
{
//    std::cout << std::fixed << utc_timepoint_to_jd() << '\n';

    const unsigned JPL_MAX_N_CONSTANTS = 1018;
    char nams[JPL_MAX_N_CONSTANTS][6], buff[102];
    double vals[JPL_MAX_N_CONSTANTS];

    void* ephem = jpl_init_ephemeris( "/Users/stevenlovegrove/code/telescope/data/de440/linux_p1550p2650.440", nams, vals);
    const double start_jd = jpl_get_double( ephem, JPL_EPHEM_START_JD);
    const double end_jd = jpl_get_double( ephem, JPL_EPHEM_END_JD);

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

    pangolin::Var<double> ui_jd("ui.JD", now_jd, now_jd, now_plus_1y_jd);

    while( !pangolin::ShouldQuit() )
    {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if(d_cam.IsShown()) {
          d_cam.Activate(s_cam);
          RenderSolarSystem(ephem, ui_jd);
      }

      pangolin::FinishFrame();
    }

    jpl_close_ephemeris( ephem);
}

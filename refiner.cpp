#include "refiner.h"
#include "qx_upsampling/qx_zdepth_upsampling_using_rbf.h"
#include "../Qing/qing_matching_cost.h"
#include "../Qing/qing_disp.h"
#include "../Qing/qing_ply.h"
#include "../Qing/qing_file_reader.h"
#include "../Qing/qing_dir.h"
#include "../Qing/qing_image.h"

void qing_get_min_max_disp(vector<float>& disp, float& min_val, float& max_val){
    min_val = 1000000.0f;
    max_val = disp[0];
    for(int i = 0, size = disp.size(); i < size; ++i) {
        if(disp[i]==0.f) continue;
        if(disp[i] > max_val) {max_val = disp[i];}
        if(disp[i] < min_val) {min_val = disp[i];}
    }
}

StereoRefiner::StereoRefiner(string frame_folder, string mask_folder,  string stereo_folder,  string scanner_folder, string frame_name,  string stereo_fn)
{
    stereo_fn = stereo_folder + frame_name + "/" + stereo_fn;
    cout << stereo_fn << endl;
    fstream fin(stereo_fn, ios::in);
    if(fin.is_open() == false) {
        cerr << "failed to open " << stereo_fn << endl;
        exit(0);
    }
    string buffer;
    fin >> m_stereo_name;
    fin >> m_stereo_id;
    fin >> buffer >> buffer;
    fin >> m_frame_name;

    m_out_dir = "../" + m_frame_name;
    qing_create_dir(m_out_dir);
    m_out_dir = m_out_dir  + '/' + m_stereo_name;
    qing_create_dir(m_out_dir);

    string filename_l, filename_r;
    string maskname_l, maskname_r;
    fin >> filename_l >> filename_r >> maskname_l >> maskname_r;
    fin >> m_crop_point_l.x >> m_crop_point_l.y;
    fin >> m_crop_point_r.x >> m_crop_point_r.y;
    fin >> m_crop_w >> m_crop_h;
    fin >> m_max_disp;
    fin >> m_min_disp;
    m_qmatrix = Mat::zeros(4,4,CV_64FC1);
    double * ptr = m_qmatrix.ptr<double>(0);
    for(int i = 0; i < 16; ++i) {
        fin >> ptr[i];
    }
    fin.close();
    cout << m_qmatrix << endl;

    filename_l = frame_folder + frame_name + "/" + filename_l;
    filename_r = frame_folder + frame_name + "/" + filename_r;
    maskname_l = mask_folder  + frame_name + "/" + maskname_l;
    maskname_r = mask_folder  + frame_name + "/" + maskname_r;

    m_raw_view_l = imread(filename_l, CV_LOAD_IMAGE_UNCHANGED);
    m_raw_view_r = imread(filename_r, CV_LOAD_IMAGE_UNCHANGED);
    m_raw_mask_l = imread(maskname_l, CV_LOAD_IMAGE_GRAYSCALE);
    m_raw_mask_r = imread(maskname_r, CV_LOAD_IMAGE_GRAYSCALE);
    threshold(m_raw_mask_l, m_raw_mask_l, 125, 255, CV_THRESH_BINARY);
    threshold(m_raw_mask_r, m_raw_mask_r, 125, 255, CV_THRESH_BINARY);

    if(0==m_raw_view_l.data || 0==m_raw_view_r.data || 0==m_raw_mask_l.data || 0==m_raw_mask_r.data) {
        cerr << "failed to load images or masks" << endl;
        cout << filename_l << endl;
        cout << filename_r << endl;
        cout << maskname_l << endl;
        cout << maskname_r << endl;
    }

    m_view_l.create(m_crop_h, m_crop_w, CV_8UC3);
    m_view_r.create(m_crop_h, m_crop_w, CV_8UC3);
    m_mask_l.create(m_crop_h, m_crop_w, CV_8UC1);
    m_mask_r.create(m_crop_h, m_crop_w, CV_8UC1);
    m_raw_view_l(Rect(m_crop_point_l, Size(m_crop_w, m_crop_h))).copyTo(m_view_l);
    m_raw_view_r(Rect(m_crop_point_r, Size(m_crop_w, m_crop_h))).copyTo(m_view_r);
    m_raw_mask_l(Rect(m_crop_point_l, Size(m_crop_w, m_crop_h))).copyTo(m_mask_l);
    m_raw_mask_r(Rect(m_crop_point_r, Size(m_crop_w, m_crop_h))).copyTo(m_mask_r);

    m_crop_size = m_crop_h * m_crop_w;
# if 1
    Mat temp_l, temp_r ;
    m_view_l.copyTo(temp_l, m_mask_l);
    imwrite(m_out_dir + "/with_mask_crop_view_l.png", temp_l);
    m_view_r.copyTo(temp_r, m_mask_r);
    imwrite(m_out_dir + "/with_mask_crop_view_r.png", temp_r);
# endif

    cvtColor(m_view_l, m_gray_l, CV_BGR2GRAY);
    cvtColor(m_view_r, m_gray_r, CV_BGR2GRAY);

    m_scanner_dir = scanner_folder + frame_name + "/" + m_stereo_name;
    cout << m_scanner_dir << endl;
}

void StereoRefiner::init_params() {
    m_disp_step = 1;
    m_wnd = 10;
    m_patch_size = m_crop_h * 0.1;
    m_patch_overlap = m_wnd;
    cout << "params: disp_step = " << m_disp_step << "\twnd = " << m_wnd << "\tpatch_size =" << m_patch_size << endl;
}


void StereoRefiner::read_in_scanner_results(const int level) {
    string disp_name_l = m_scanner_dir + "/final_disp_l_" + qing_int_2_string(level) + ".txt";
    string disp_name_r = m_scanner_dir + "/final_disp_r_" + qing_int_2_string(level) + ".txt";

    float temp_min_disp, temp_max_disp;

    if(level != 0) {
        string view_name_l = m_scanner_dir + "/crop_imgL_" + qing_int_2_string(level) + ".jpg";
        string view_name_r = m_scanner_dir + "/crop_imgR_" + qing_int_2_string(level) + ".jpg";
        string mask_name_l = m_scanner_dir + "/crop_mskL_" + qing_int_2_string(level) + ".jpg";
        string mask_name_r = m_scanner_dir + "/crop_mskR_" + qing_int_2_string(level) + ".jpg";

        m_view_l = imread(view_name_l, 1);
        m_view_r = imread(view_name_r, 1);
        m_mask_l = imread(mask_name_l, 0);   cv::threshold(m_mask_l, m_mask_l, 125, 255, THRESH_BINARY);
        m_mask_r = imread(mask_name_r, 0);   cv::threshold(m_mask_r, m_mask_r, 125, 255, THRESH_BINARY);

        m_crop_w = m_view_l.size().width;
        m_crop_h = m_view_l.size().height;
    }
    else
    {
        cout << m_view_l.size() << endl;
        cout << m_view_r.size() << endl;
        cout << m_mask_l.size() << endl;
        cout << m_mask_r.size() << endl;
    }

    m_disp_l.clear();
    m_disp_r.clear();
    qing_read_disp_txt(disp_name_l, m_crop_h, m_crop_w, m_disp_l);
    qing_read_disp_txt(disp_name_r, m_crop_h, m_crop_w, m_disp_r);
    qing_get_min_max_disp(m_disp_l, m_min_disp,  m_max_disp);
    qing_get_min_max_disp(m_disp_r, temp_min_disp,  temp_max_disp);
    m_min_disp = min(temp_min_disp, m_min_disp);
    m_max_disp = max(temp_max_disp, m_max_disp);

    m_d_range = (m_max_disp - m_min_disp) / m_disp_step + 1;

    cout << "level " << level << ": " << m_view_l.size() << "\t" << m_min_disp << " ~ " << m_max_disp << endl;
    cout << "read in scanner results in level " << level << " done..." << endl;
}

void StereoRefiner::sgbm_refine(const int level) {
    int cnt = 0, start_y, end_y;
    //   int vol_total_size = m_crop_w * m_patch_size * m_d_range; cout << "cost vol size of a patch: " << vol_total_size << endl;

    StereoSGBM sgbm;
    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = 3;

    int cn = 3, h, w;                               //image channels
    double minVal, maxVal;

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;										 //0
    sgbm.numberOfDisparities = 560;                              //total search disparity : 480 * 16
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 100;
    sgbm.speckleRange = 32;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = true;

    Mat patch_l, patch_r, patch_disp;
    Mat temp_refine_disp = Mat::zeros(m_crop_h, m_crop_w, CV_16SC1), temp_median_disp;
    Mat temp_show_disp;
    m_refine_disp = Mat::zeros(m_crop_h, m_crop_w, CV_32FC1);
    m_show_disp = Mat::zeros(m_crop_h, m_crop_w, CV_8U);

    for(int y = 0; y < m_crop_h; ) {
        start_y = y;
        end_y = min(start_y + m_patch_size, m_crop_h);
        cout << cnt++ << ": " << start_y << " ~ " << end_y << '\t';                 //[start_y, end_y)
        y += m_patch_size - m_patch_overlap;

        h = end_y - start_y ;
        w = m_crop_w;

        patch_l=Mat::zeros(h,w,m_view_l.type());
        patch_r=Mat::zeros(h,w,m_view_r.type());
        m_view_l(Rect(0, start_y, w, h)).copyTo(patch_l);
        m_view_r(Rect(0, start_y, w, h)).copyTo(patch_r);

        sgbm(patch_l, patch_r, patch_disp);


        minMaxIdx(patch_disp, &minVal, &maxVal);
        cout << minVal << "~" << maxVal << "\t";

        patch_disp.copyTo(temp_refine_disp(Rect(0,start_y,w,h)));
        cout << "sgbm done.." << endl;
    }

    qing_call_qx_rbf(temp_refine_disp);
    medianBlur(temp_refine_disp, temp_median_disp, 5);

    string filename  ;

    temp_refine_disp.convertTo(m_refine_disp, CV_32FC1, 1.0f/16);
    temp_refine_disp.convertTo(temp_show_disp, CV_8U, 255/(sgbm.numberOfDisparities*16.0));
    temp_show_disp.copyTo(m_show_disp, m_mask_l);
    filename = m_out_dir + "/sgbm_disp_" + qing_int_2_string(level) + ".png";
    imwrite(filename, m_show_disp); cout << "saving " << filename << endl;

    Mat erode_mask = qing_erode_image(m_mask_l, 20);
    m_show_disp=Mat::zeros(m_show_disp.size(), m_show_disp.type());
    temp_show_disp.copyTo(m_show_disp, erode_mask);
    filename = m_out_dir + "/erode_sgbm_disp_" + qing_int_2_string(level) + ".png";
    imwrite(filename, m_show_disp); cout << "saving " << filename << endl;

    temp_median_disp.convertTo(m_median_refine_disp, CV_32FC1, 1.0f/16);
    temp_median_disp.convertTo(temp_show_disp, CV_8U, 255/(sgbm.numberOfDisparities*16.0));
    temp_show_disp.copyTo(m_median_show_disp, m_mask_l);
    filename = m_out_dir + "/median_sgbm_disp_" + qing_int_2_string(level) + ".png";
    imwrite(filename, m_median_show_disp); cout << "saving " << filename << endl;

    m_median_show_disp=Mat::zeros(m_median_show_disp.size(), m_median_show_disp.type());
    temp_show_disp.copyTo(m_median_show_disp, erode_mask);
    filename = m_out_dir + "/erode_median_sgbm_disp_" + qing_int_2_string(level) + ".png";
    imwrite(filename, m_median_show_disp); cout << "saving " << filename << endl;
}

void StereoRefiner::qing_call_qx_rbf(Mat& disp)
{
    float sigma_spatial = 0.005f;
    float sigma_range = 0.1f;

    int m_w = disp.size().width;
    int m_h = disp.size().height;
    int m_total = m_h * m_w;

    cout << m_w << " x " << m_h << ", m_total = " << m_total  << ", nchannel = " << disp.channels() << endl;

    Mat float_disp = Mat::zeros(m_h, m_w, CV_32FC1);
    disp.convertTo(float_disp, CV_32FC1);               //type of disp: CV_16SC1

    vector<uchar> guidance(m_total*3);
    vector<uchar> gradient_x(m_total), gradient_y(m_total);                                                                                     //color image gradients
    vector<float> out(m_total), in(m_total), temp(m_total), temp_2w(2*m_w), ones(m_total), ones_temp(m_total), ones_temp_2w(2*m_w);             //disparity

    vector<float> float_disp_vec(m_total);
    vector<uchar> mask_vec(m_total);
    memcpy(&float_disp_vec.front(), float_disp.data, m_total*sizeof(float));
    memcpy(&mask_vec.front(), m_mask_l.data, m_total*sizeof(uchar));
    memcpy(&guidance.front(), m_view_l.data, sizeof(uchar)*m_total*3);
    copy(float_disp_vec.begin(), float_disp_vec.end(), in.begin());
    for(int idx = 0; idx < m_total; ++idx) {
        if(0==mask_vec[idx]) continue;
        if(5.f > in[idx]) in[idx] = 0.f;
    }
    Mat test_disp(m_h, m_w, CV_32FC1);
    memcpy(test_disp.data, &in.front(), m_total * sizeof(float));
    qx_zdepth_upsampling_using_rbf(&out[0], &in[0], &guidance[0], &temp[0], &temp_2w[0], &gradient_x[0], &gradient_y[0],
            &ones[0], &ones_temp[0], &ones_temp_2w[0], m_h, m_w, sigma_spatial, sigma_range);

    for(int idx = 0; idx < m_total; idx++) {
        if(0==mask_vec[idx]) float_disp_vec[idx] = 0.f; //continue;
        else
        {
            float_disp_vec[idx] = out[idx];
        }
    }

    memcpy(float_disp.data, &float_disp_vec.front(), m_total*sizeof(float));
    float_disp.copyTo(disp, m_mask_l);
}


void StereoRefiner::triangulate(const int level) {
    cout << "\ntriangulate 3d points from disparity results..." ;

    int erode_size = (!level)? 20:10;
    Mat erode_mask = qing_erode_image(m_mask_l, erode_size);

    float * ptr_disp = (float *)m_refine_disp.ptr<float>(0);
    unsigned char * ptr_msk  = (unsigned char *)erode_mask.ptr<unsigned char>(0);

    double minVal, maxVal;
    minMaxIdx(m_refine_disp, &minVal, &maxVal);
    cout << "max_d = " << maxVal << ", min_d = " << minVal << '\t';

    int scale = 1 << level;
    cout << "scale = " << scale << "\t";

    //preparing disparity
    float offset = (m_crop_point_l.x - m_crop_point_r.x) / (scale*1.f);
    cout << "offset = " << offset << endl;
    for(int y = 0, index = 0; y < m_crop_h; ++y) {
        for(int x = 0; x < m_crop_w; ++x) {
            if( 255 == ptr_msk[index] && 0 < ptr_disp[index] )  ptr_disp[index] += offset;
            else  ptr_disp[index] = 0.f;
            index ++;
        }
    }

    vector<Vec3f> points; points.reserve(m_crop_size);
    vector<Vec3f> colors; colors.reserve(m_crop_size);

    unsigned char * ptr_bgr  = (unsigned char *)m_view_l.ptr<unsigned char>(0);
    double * ptr_q_matrix = new double[16];
    memcpy(ptr_q_matrix,  (double *)m_qmatrix.ptr<double>(0), 16 * sizeof(double));
    ptr_q_matrix[3]  /= scale; //0 + 3
    ptr_q_matrix[7]  /= scale; //4 + 3
    ptr_q_matrix[11] /= scale; //8 + 3

    qing_disp_2_depth(points, colors, ptr_disp, ptr_msk, ptr_bgr, ptr_q_matrix, m_crop_point_l, m_crop_w, m_crop_h);
    string savefn = m_out_dir + "/" + qing_int_2_format_string(m_stereo_id, 2, '0') + "_" +  m_frame_name + "_pointcloud_" + m_stereo_name + "_" + qing_int_2_string(level) +  ".ply";
    qing_write_point_color_ply(savefn, points, colors);
    cout << "save " << savefn << " done. " << points.size() << " Points." << m_crop_w << '\t' << m_crop_h << endl;
}


//void StereoRefiner::census_transform() {

//    m_census_l.resize(m_crop_size);
//    m_census_r.resize(m_crop_size);
//    qx_census_transform_3x3(&m_census_l.front(), m_gray_l.data, m_crop_h, m_crop_w);
//    cout << "end of census transform left" << endl;
//    qx_census_transform_3x3(&m_census_r.front(), m_gray_r.data, m_crop_h, m_crop_w);
//    cout << "end of census transform right" << endl;

//# if 1
//    Mat census_l(m_crop_h, m_crop_w, CV_8UC1);
//    Mat census_r(m_crop_h, m_crop_w, CV_8UC1);
//    memcpy(census_l.data, &m_census_l.front(), m_crop_size * sizeof(unsigned char));
//    memcpy(census_r.data, &m_census_r.front(), m_crop_size * sizeof(unsigned char));
//    imwrite(m_out_dir + "/census_l.jpg", census_l);
//    imwrite(m_out_dir + "/census_r.jpg", census_r);
//# endif
//}





























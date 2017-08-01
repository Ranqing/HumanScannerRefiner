#include "refiner.h"

int main(int argc, char * argv[])
{
    cout << "usage: "
         << argv[0]
         << "  /media/ranqing/Work/ZJU/20170618/Humans_frames/\t"
         << "  /media/ranqing/Work/ZJU/20170618/Humans_masks/\t"
         << "  /media/ranqing/Work/ZJU/20170618/Humans_stereos/\t"
         << "  /media/ranqing/Work/ZJU/20170618/Humans_results/\t"
         << "  FRM_0245\t stereo_A01A02.info" << endl;




    if(argc != 7)
    {
        cerr << "invalid arguments.." << endl;
        return -1;
    }

    string img_folder  = argv[1];
    string msk_folder  = argv[2] ;
    string info_folder = argv[3];
    string scanner_folder = argv[4];
    string frame_name = argv[5];
    string stereo_fn  = argv[6];

    StereoRefiner * refiner = new StereoRefiner(img_folder, msk_folder, info_folder, scanner_folder, frame_name, stereo_fn);
    refiner->init_params();

    int st = 0;
    for(int level = st; level >= 0; level--)
    {
        refiner->sgbm_refine(level);
        refiner->triangulate(level);
    }

    return 1;
}

import os
import argparse

def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu', default = '0', type = str, help = 'gpu ids, -1 for cpu, default is 0.')
    parser.add_argument('-d','--dataroot', default = './examples', type = str, help = 'the input folder that contains test face photos, default is ./examples')
    parser.add_argument('-s','--savefolder', default = '3styles', type = str, help = 'the name of save folder that contains result images, default is 3styles')
    return parser.parse_args()

if __name__ == '__main__':
    opt = opts()
    exp = 'pretrained'
    imgsize = 512
    epoch = '200'
    dataroot = opt.dataroot
    gpu_id = opt.gpu

    # test 3 styles in one pass
    savefolder = 'images'+opt.savefolder
    os.system('python3 test.py --dataroot %s --name %s --model test_3styles --output_nc 1 --no_dropout --num_test 1000 --epoch %s --imagefolder %s --crop_size %d --load_size %d --gpu_ids %s' % (dataroot,exp,epoch,savefolder,imgsize,imgsize,gpu_id))
    print('check ./results/%s/test_%s/index%s.html'%(exp,epoch,savefolder[6:]))
    print('saved to ./results/%s/test_%s/%s'%(exp,epoch,savefolder))

    # test 3 styles separately
    '''
    for vec in [[1,0,0],[0,1,0],[0,0,1]]:
        #1,0,0 for style1; 0,1,0 for style2; 0,0,1 for style3
        svec = '%d,%d,%d' % (vec[0],vec[1],vec[2])
        savefolder = 'imagesstyle%d-%d-%d'%(vec[0],vec[1],vec[2])
        print('results/%s/test_%s/index%s.html'%(exp,epoch,savefolder[6:]))
        os.system('python3 test.py --dataroot %s --name %s --model test --output_nc 1 --no_dropout --model_suffix _A --num_test 1000 --epoch %s --imagefolder %s --sinput svec --svec %s --crop_size %d --load_size %d --gpu_ids %s' % (dataroot,exp,epoch,savefolder,svec,imgsize,imgsize,gpu_id))
    '''
        
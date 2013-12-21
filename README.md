Video Segmentation by MRF Propagation
===========================

If using this code please cite: 


Efficient MRF Energy Propagation for Video Segmentation via Bilateral Filters.
Ozan Sener, Kemal Ugur, Aydin Alatan
Submitted to Transactions on Multimedia arXiv:1301.5356v1


============================================

Dependencies:


* OpenCV version 2.4 or greater (dev version or install from source)
* Boost version 1.50 or greater

----

Commands to install the required dependencies

    # install opencv
    sudo apt-get install libopencv-dev
    #install boost 1.50
    wget http://sourceforge.net/projects/boost/files/boost/1.50.0/boost_1_50_0.tar.bz2
    tar --bzip2 -xf boost_1_50_0.tar.bz2
    #If you prefer to install boost to a specific directory use the following instead
    # ./bootstrap.sh --prefix=path/to/installation/prefix
    ./bootstrap.sh
    ./b2
    sudo ./b2 install
    
    # download code 
    git clone git@bitbucket.org:ozansener/videosegmentationbiexp.git
    
    # compile
    cd videosegmentationbiexp
    mkdir Build
    cd Build
    cmake ..
    make
    
    # run segmentation 
    cd ../build/
    ./segmenter
    

--------------







/* Global algorithmic parameters */

// modes 
#ifdef VIDEO_MODE
#define CAMERA_INDEX        (0)
#else 
#define IMAGE_PATH          "C:/Users/Alfred/Desktop/SwapFace/testData/3.jpg"
#endif

// resources 
#define CASCADE_PATH        "C:/Users/Alfred/Desktop/SwapFace/res/haarcascade_frontalface_default.xml"

// debug parameters 
#define VISUALIZATION	    (1)

// resize parameters 
#define WIDTH               (640)
#define HEIGHT			    (480)

// algorithm parameters 
#define PYRAMID_DEPTH       (4)

#define SCALE_FACTOR        (1.1)
#define MIN_NEIGHBOURS      (2)
#define WINDOW_SIZE         (100)

#define CLUSTER_COUNT       (2)
#define CLUSTER_ATTEMPTS    (1)

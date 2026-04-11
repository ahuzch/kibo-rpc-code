package jp.jaxa.iss.kibo.rpc.sampleapk;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import gov.nasa.arc.astrobee.Result;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.aruco.Aruco;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.core.Mat;
//import org.opencv.core.Rect;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;

/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */
public class YourService extends KiboRpcService {
    private final String TAG = this.getClass().getSimpleName();
    private final int LOOPMAX = 3;

    //Template file name
    private final String[] TEMPLATE_FILE_NAME = {
            "coin.png",
            "compass.png",
            "coral.png",
            "crystal.png",
            "diamond.png",
            "emerald.png",
            "fossil.png",
            "key.png",
            "letter.png",
            "shell.png",
            "treasure_box.png"
    };

    //Template name
    private final String[] TEMPLATE_NAME = {
            "coin",
            "compass",
            "coral",
            "crystal",
            "diamond",
            "emerald",
            "fossil",
            "key",
            "letter",
            "shell",
            "treasure_box"
    };

    private static final float MARKER_SIZE = 0.05f; // 5cm - actual aruco marker size
    private static final double THRESHOLD_DISTANCE = 0.5; // Target distance in meters
    private static final double MOVEMENT_FACTOR = 0.7; // Moves x% of the distance, ideally dont go above 0.5 as it then position is outside KIZ 1
    /*
    private static final int CROP_WIDTH = 300;  // pixels
    private static final int CROP_HEIGHT = 300; // pixels
    private static final double CROP_CENTER_X_RATIO = 0.7; // 70% of image width
    private static final double CROP_CENTER_Y_RATIO = 0.7; // 70% of image height
    */

    @Override
    protected void runPlan1() {
        // The mission starts.
        api.startMission();
        Log.i(TAG, "Start Mission");


        // Move to kiz2
        Point point1 = new Point(10.5d, -9.9d, 4.6d);
        Quaternion quaternion1 = new Quaternion(0f, 0f, 0f, 1f);
        api.moveTo(point1, quaternion1, true);

	    //go to area 1
        Point point2 = new Point(10.95d, -10.12d ,5.195d);
        Quaternion quaternion2 = new Quaternion(0f, 0f, -0.707f, 0.707f);
        api.moveTo(point2, quaternion2, true);

        Mat image1 = api.getMatNavCam();
        api.saveMatImage(image1, "Area1.jpg");
        detectImage(image1, 1);
	

	    //go to area 2
        Point point3 = new Point(10.925d, -8.875d, 4.32d);
        Quaternion quaternion3 = new Quaternion(0f, 0.707f, 0f, 0.707f);
        api.moveTo(point3, quaternion3, true);

        Mat image2 = api.getMatNavCam();
        api.saveMatImage(image2, "Area2.jpg");
        detectImage(image2, 2);
	
	    //go to area 3
        Point point4 = new Point(10.925d, -7.925d, 4.32d);
        Quaternion quaternion4 = new Quaternion(0f, 0.707f, 0f, 0.707f);
        api.moveTo(point4, quaternion4, true);

        Mat image3 = api.getMatNavCam();
        api.saveMatImage(image3, "Area3.jpg");
        detectImage(image3, 3);
	
	    //go to area 4
        Point point5 = new Point(10.38d, -6.8525d, 4.945d);
        Quaternion quaternion5 = new Quaternion(0f, 0f, 0f, 1f);
        api.moveTo(point5, quaternion5, true);

        Mat image4 = api.getMatDockCam();
        api.saveMatImage(image4, "Area4.jpg");
        detectImage(image4,4);

	
        // Fly to astronaut, report completion, and snapshot
        Point point6 = new Point(11.143d, -6.7607d, 4.9654d);
        Quaternion quaternion6 = new Quaternion(0f, 0f, 0.707f, 0.707f);
        api.moveTo(point6, quaternion6, false);
        api.reportRoundingCompletion();
        api.notifyRecognitionItem();
        api.takeTargetItemSnapshot();

    }

    @Override
    protected void runPlan2() {
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3() {
        // write your plan 3 here.
    }

    /**
     * Detect items in an image and report them to the Kibo API
     * @param image The image to analyze
     * @param area The area number (1-4)
     */
    private void detectImage(Mat image, int area) {
        final long timeout = 5000; // 5 seconds
        final long startTime = System.currentTimeMillis();
        
        // Declare all Mats outside try block for finally access
        Mat undistorted = null;
        Mat cameraMatrix = null;
        Mat distCoeffs = null;
        Mat rvecs = null;
        Mat tvecs = null;
        
        
        // Initialize camera parameters
        cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);
        distCoeffs = new Mat(1, 5, CvType.CV_64F);
        distCoeffs.put(0, 0, api.getNavCamIntrinsics()[1]);
        distCoeffs.convertTo(distCoeffs, CvType.CV_64F);

        // Undistort frame
        undistorted = new Mat();
        Calib3d.undistort(image, undistorted, cameraMatrix, distCoeffs);

        // AR Marker Detection
        Dictionary dictionary = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
        List<Mat> corners = new ArrayList<>();
        Mat markerIds = new Mat();
        Aruco.detectMarkers(undistorted, dictionary, corners, markerIds);

        try{
            if (!markerIds.empty()) {
                Log.i(TAG, "Detected AR markers in area " + area + ":");
                
                // Pose estimation
                rvecs = new Mat();
                tvecs = new Mat();
                Aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);
                
                for (int i = 0; i < markerIds.rows(); i++) {
                    double id = markerIds.get(i, 0)[0];
                    double[] tvec = new double[3];
                    tvecs.get(i, 0, tvec);
                    
                    double distance = Math.sqrt(tvec[0]*tvec[0] + tvec[1]*tvec[1] + tvec[2]*tvec[2]);
                    Log.i(TAG, String.format("Marker ID: %d Distance=%.2fm X=%.2fm Y=%.2fm Z=%.2fm",
                        (int)id, distance, tvec[0], tvec[1], tvec[2]));
                    
                    if (distance > THRESHOLD_DISTANCE) {
                        Log.i(TAG, String.format("MARKER TOO FAR | Current: %.2fm > Threshold: %.2fm | Approaching...",distance, THRESHOLD_DISTANCE));
                        approachMarker(tvec);
                        Mat newImage = api.getMatNavCam();
                        api.saveMatImage(newImage, "AR_image_area" + area + ".jpg");
                        try {
                            performTemplateMatching(newImage, area);
                        } finally {
                            newImage.release();
                        }
                        return;
                    }
                    else{
                        Log.i(TAG, String.format("MARKER WITHIN RANGE | Current: %.2fm ≤ Threshold: %.2fm | Proceeding with template matching",distance, THRESHOLD_DISTANCE));
                    }
                }
            }
            
            performTemplateMatching(undistorted, area);
            
        } catch (Exception e) {
            Log.e(TAG, "Detection failed", e);
        } finally {
            // Release all Mats
            if (undistorted != null) undistorted.release();
            if (cameraMatrix != null) cameraMatrix.release();
            if (distCoeffs != null) distCoeffs.release();
            if (rvecs != null) rvecs.release();
            if (tvecs != null) tvecs.release();
        }
    }

    private void approachMarker(double[] tvec) {
        // KIZ 1 boundaries
        final double X_MIN = 10.3, X_MAX = 11.55;
        final double Y_MIN = -10.2, Y_MAX = -6.0;
        final double Z_MIN = 4.32, Z_MAX = 5.57;

        // 1. Get current position
        Point currentPos = api.getRobotKinematics().getPosition();
        Quaternion currentQuat = api.getRobotKinematics().getOrientation();
        
        // 2. Calculate movement vector 
        double moveX = tvec[0] * MOVEMENT_FACTOR;
        double moveY = tvec[1] * MOVEMENT_FACTOR;
        double moveZ = tvec[2] * MOVEMENT_FACTOR;
        
        // 3. Calculate new target position
        double proposedX = currentPos.getX() - moveX;
        double proposedY = currentPos.getY() + moveY;
        double proposedZ = currentPos.getZ() - moveZ;

        // 4. Apply boundary constraints
        double clampedX = Math.max(X_MIN, Math.min(X_MAX, proposedX));
        double clampedY = Math.max(Y_MIN, Math.min(Y_MAX, proposedY));
        double clampedZ = Math.max(Z_MIN, Math.min(Z_MAX, proposedZ));

        // 5. Create validated target position
        Point targetPos = new Point(clampedX, clampedY, clampedZ);

        // 6. Log adjustments if needed
        if (proposedX != clampedX || proposedY != clampedY || proposedZ != clampedZ) {
            Log.w(TAG, String.format(
                "ADJUSTED MOVEMENT TO STAY IN KIZ | Original: (%.2f, %.2f, %.2f) → Clamped: (%.2f, %.2f, %.2f)", proposedX, proposedY, proposedZ, clampedX, clampedY, clampedZ));
        } else{
            Log.w(TAG,String.format("Clamped Position is within KIZ"));
        }
        
        Log.i(TAG, String.format("Approaching marker: Moving by (%.2f, %.2f, %.2f)", moveX, moveY, moveZ));
        
        // 7. Execute movement
        api.moveTo(targetPos, currentQuat, false);
    }

    private void performTemplateMatching(Mat image, int area) {
        // Validate input
        if (image.empty()) {
            Log.e(TAG, "Empty image provided for template matching");
            return;
        }

        Mat undistortImg = new Mat();
        //Mat croppedImg = null;
        //Mat processingImg = null;

        // Load template images
        Mat[] templates = new Mat[TEMPLATE_FILE_NAME.length];
        // Number of matches for each template
        int[] templateMatchCnt = new int[TEMPLATE_FILE_NAME.length];

        try {
            // 1. Undistort image (using fresh camera parameters)
            Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
            cameraMatrix.put(0, 0, api.getNavCamIntrinsics()[0]);
            Mat distCoeffs = new Mat(1, 5, CvType.CV_64F);
            distCoeffs.put(0, 0, api.getNavCamIntrinsics()[1]);
            Calib3d.undistort(image, undistortImg, cameraMatrix, distCoeffs);

            // Crop the undistorted image to focus on center
            //croppedImg = cropCenter(undistortImg);
            //processingImg = croppedImg.clone(); // We'll use this for matching
            //api.saveMatImage(processingImg, "cropped_image_area" + area + ".jpg");

            // Pattern matching
            for (int i = 0; i < TEMPLATE_FILE_NAME.length; i++) {
                try {
                    // Open the template image file in the Bitmap from the file name and convert to Mat
                    InputStream inputStream = getAssets().open(TEMPLATE_FILE_NAME[i]);
                    Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                    Mat mat = new Mat();
                    Utils.bitmapToMat(bitmap, mat);

                    // Convert to grayscale
                    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);

                    // Assign to an array of templates
                    templates[i] = mat;

                    inputStream.close();
                } catch (IOException e) {
                    Log.e(TAG, "Error loading template: " + TEMPLATE_FILE_NAME[i], e);
                }
            }

            // Get the number of template matches
            for (int tempNum = 0; tempNum < templates.length; tempNum++) {
                // Number of matches
                int matchCnt = 0;
                // Coordinates of the matched location
                List<org.opencv.core.Point> matches = new ArrayList<>();

                // Declare Mats outside try block for cleanup
                Mat template = null;
                Mat targetImg = null;
                
                try {
                    // Loading template image and target image
                    template = templates[tempNum].clone();
                    targetImg = undistortImg.clone();

                    // Pattern matching parameters
                    int widthMin = 40;
                    int widthMax = 80;
                    int changeWidth = 10;
                    int changeAngle = 90;

                    for (int i = widthMin; i <= widthMax; i += changeWidth) {
                        for (int j = 0; j < 360; j += changeAngle) {
                            // Declare temporary Mats for this iteration
                            Mat resizedTemp = null;
                            Mat rotResizedTemp = null;
                            Mat result = null;
                            Mat thresholdedResult = null;
                            
                            try {
                                resizedTemp = resizeImg(template, i);
                                rotResizedTemp = rotImg(resizedTemp, j);
                                result = new Mat();
                                
                                Imgproc.matchTemplate(targetImg, rotResizedTemp, result, Imgproc.TM_CCOEFF_NORMED);

                                // Get coordinates with similarities greater than or equal to the threshold
                                double threshold = 0.8;
                                Core.MinMaxLocResult mmlr = Core.minMaxLoc(result);
                                double maxVal = mmlr.maxVal;
                                if (maxVal >= threshold) {
                                    thresholdedResult = new Mat();
                                    Imgproc.threshold(result, thresholdedResult, threshold, 1.0, Imgproc.THRESH_TOZERO);

                                    // Get match counts
                                    for (int y = 0; y < thresholdedResult.rows(); y++) {
                                        for (int x = 0; x < thresholdedResult.cols(); x++) {
                                            if (thresholdedResult.get(y, x)[0] > 0) {
                                                matches.add(new org.opencv.core.Point(x, y));
                                            }
                                        }
                                    }
                                }
                            } finally {
                                // Release temporary Mats for this scale/rotation
                                if (resizedTemp != null) resizedTemp.release();
                                if (rotResizedTemp != null) rotResizedTemp.release();
                                if (result != null) result.release();
                                if (thresholdedResult != null) thresholdedResult.release();
                            }
                        }
                    }

                    // Avoid detecting the same location multiple times
                    List<org.opencv.core.Point> filteredMatches = removeDuplicates(matches);
                    matchCnt += filteredMatches.size();

                    // Number of matches for each template
                    templateMatchCnt[tempNum] = matchCnt;
                    Log.d(TAG, "Template " + TEMPLATE_NAME[tempNum] + " matches: " + matchCnt);
                    
                } finally {
                    // Release Mats for this template
                    if (template != null) template.release();
                    if (targetImg != null) targetImg.release();
                }
            }
            // When you recognize landmark items, set the type and number
            int mostMatchTemplateNum = getMaxIndex(templateMatchCnt);
            if (templateMatchCnt[mostMatchTemplateNum] > 0) {
                api.setAreaInfo(area, TEMPLATE_NAME[mostMatchTemplateNum], templateMatchCnt[mostMatchTemplateNum]);
                Log.i(TAG, "Detected in area " + area + ": " + TEMPLATE_NAME[mostMatchTemplateNum] + 
                    " (count: " + templateMatchCnt[mostMatchTemplateNum] + ")");
            } else {
                Log.w(TAG, "No items detected in area " + area);
            }
        } finally {
            // Clean up all resources
            if (undistortImg != null) undistortImg.release();

            // Clean up cropped images if they exist
            //if (croppedImg != null) croppedImg.release();
            //if (processingImg != null) processingImg.release();
            
            // Clean up templates
            for (Mat template : templates) {
                if (template != null) template.release();
            }
        }
    }
    
    /*
    private Mat cropCenter(Mat image) {
        // Calculate crop coordinates
        int centerX = (int)(image.cols() * CROP_CENTER_X_RATIO);
        int centerY = (int)(image.rows() * CROP_CENTER_Y_RATIO);
        int x = Math.max(0, centerX - CROP_WIDTH/2);
        int y = Math.max(0, centerY - CROP_HEIGHT/2);
        int width = Math.min(CROP_WIDTH, image.cols() - x);
        int height = Math.min(CROP_HEIGHT, image.rows() - y);
        
        // Perform the crop
        Rect cropRect = new Rect(x, y, width, height);
        Mat cropped = new Mat(image, cropRect);
        
        Log.d(TAG, String.format("Cropped image: %dx%d from (%d,%d)", width, height, x, y));
        return cropped;
    }
    */
    // Resize image
    private Mat resizeImg(Mat img, int width) {
        int height = (int) (img.rows() * ((double) width / img.cols()));
        Mat resizedImg = new Mat();
        Imgproc.resize(img, resizedImg, new Size(width, height));
        return resizedImg;
    }

    // Rotate image
    private Mat rotImg(Mat img, int angle) {
        org.opencv.core.Point center = new org.opencv.core.Point(img.cols() / 2.0, img.rows() / 2.0);
        Mat rotatedMat = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat rotatedImg = new Mat();
        Imgproc.warpAffine(img, rotatedImg, rotatedMat, img.size());

        return rotatedImg;
    }

    // Remove multiple detections
    private static List<org.opencv.core.Point> removeDuplicates(List<org.opencv.core.Point> points) {
        double length = 10; // width 10 px
        List<org.opencv.core.Point> filteredList = new ArrayList<>();

        for (org.opencv.core.Point point : points) {
            boolean isInclude = false;
            for (org.opencv.core.Point checkPoint : filteredList) {
                double distance = calculateDistance(point, checkPoint);
                
                if (distance <= length) {
                    isInclude = true;
                    break;
                }
            }

            if (!isInclude) {
                filteredList.add(point);
            }
        }

        return filteredList;
    }

    // Find the distance between two points
    private static double calculateDistance(org.opencv.core.Point p1, org.opencv.core.Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
    }

    // Get the maximum number of an array
    private int getMaxIndex(int[] array) {
        int max = 0;
        int maxIndex = 0;

        // Find the index of the element with the largest value
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}

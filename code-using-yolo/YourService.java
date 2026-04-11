package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.lang.reflect.Array;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.aruco.*;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;
/*
import javax.vecmath.AxisAngle4f;
import javax.vecmath.Matrix3f;
import javax.vecmath.Quat4f;
import javax.vecmath.Vector3f;
*/


/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */
public class YourService extends KiboRpcService {
    private final String TAG = this.getClass().getSimpleName();
    private static final float THRESHOLD = 0.7f;
    private MatOfDouble camMatrix, distCoeffs;
    private ObjectDetector detector;

    @Override
    protected void runPlan1() {
        // Initialize camera calibration
        double[][] intrinsics = api.getNavCamIntrinsics();
        camMatrix = new MatOfDouble();
        camMatrix.fromArray(intrinsics[0]);
        distCoeffs = new MatOfDouble();
        distCoeffs.fromArray(intrinsics[1]);
        
        // Initialize TFLite detector
        initDetector();

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
        // write your plan 2 here
    }

    @Override
    protected void runPlan3() {
        // write your plan 3 here
    }

    /**
     * Initialize the TensorFlow Lite object detector
     */
    private void initDetector() {
        String modelName = "yolov8n_float32.tflite";
        try{
            if (!Arrays.asList(getAssets().list("")).contains(modelName)) {
                throw new IOException("Model file " + modelName + " not found in assets");
            }
            ObjectDetector.ObjectDetectorOptions options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setScoreThreshold(0.75f)
                    .setMaxResults(25)
                    .build();
        
            MappedByteBuffer modelByteBuffer = loadModelFile(getAssets(), modelName);
            detector = ObjectDetector.createFromBufferAndOptions(modelByteBuffer, options);
            Log.i(TAG, "TFLite detector initialized successfully");
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize TFLite detector", e);
        }
    }

    /**
     * Load TFLite model from assets
     */
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Main image detection function
     * @param image Input image from nav cam
     * @param area Area number (1-4)
     */
    private void detectImage(Mat image, int area) {
        if (image == null || image.empty()) {
            Log.e(TAG, "Empty image received for detection");
            return;
        }

        // Declare variables here (not inside try block)
        Mat undistortedImage = null;
        Mat processedImage = null;
        Bitmap imageBitmap = null;
        
        try {
            // Step 1: Undistort the image
            undistortedImage = undistort(image);
            
            // Step 2: Detect AR markers and align image
            processedImage = processWithArucoMarkers(undistortedImage);
            if (processedImage == null) {
                Log.w(TAG, "No AR markers found in area " + area);
                return;
            }
            else{
                Log.w(TAG, "Marker detected in Area: " + area);
            }
        
            // Step 3: Convert to bitmap for TFLite
            imageBitmap = matToBitmap(processedImage);
            
            // Step 4: Run object detection
            List<Detection> detections = detector.detect(TensorImage.fromBitmap(imageBitmap));
            
            // Step 5: Process and report results
            Map<String, Integer> detectedItems = processDetections(detections);

            // Report all detected items
            for (Map.Entry<String, Integer> entry : detectedItems.entrySet()) {
                api.setAreaInfo(area, entry.getKey(), entry.getValue());
                Log.i(TAG, String.format("Detected in area %d: %s (count: %d)", 
                    area, entry.getKey(), entry.getValue()));
            }
        } catch (Exception e) {
            Log.e(TAG, "Error during image detection", e);
        } finally {
            // Release resources
            if (undistortedImage != null) undistortedImage.release();
            if (processedImage != null) processedImage.release();
            if (imageBitmap != null) imageBitmap.recycle();
        }
    }

    private Mat undistort(Mat image) {
        Mat undistorted = new Mat();
        Calib3d.undistort(image, undistorted, camMatrix, distCoeffs);
        return undistorted; // Let caller handle releasing
    }

    /**
     * Process image using AR markers for alignment
     */
    private Mat processWithArucoMarkers(Mat image) {
        DetectorParameters params = DetectorParameters.create();
        Mat ids = new MatOfInt();
        List<Mat> corners = new ArrayList<>();
        Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

        Aruco.detectMarkers(image, dict, corners, ids, params);
        if (corners.isEmpty()) {
            return null;
        }

        // Define destination points for perspective transform
        org.opencv.core.Point p1 = new org.opencv.core.Point(217.5, 13.5);
        org.opencv.core.Point p2 = new org.opencv.core.Point(267.5, 13.5);
        org.opencv.core.Point p3 = new org.opencv.core.Point(267.5, 63.5);
        MatOfPoint2f dest = new MatOfPoint2f(p1, p2, p3);

        // Get source points from detected marker
        Mat cornersMat = corners.get(0);
        List<org.opencv.core.Point> srcPoints = new ArrayList<>();
        Converters.Mat_to_vector_Point2f(cornersMat.t(), srcPoints);
        srcPoints.remove(3); // We only need 3 points for affine transform
        MatOfPoint2f src = new MatOfPoint2f();
        src.fromList(srcPoints);

        // Apply perspective correction
        Mat alignedImage = new Mat();
        Mat affineTransform = Imgproc.getAffineTransform(src, dest);
        Imgproc.warpAffine(image, alignedImage, affineTransform, new Size(290, 170));
        
        // Crop to final size
        return alignedImage.submat(0, 170, 0, 220);
    }

    /**
     * Convert OpenCV Mat to Android Bitmap
     */
    private Bitmap matToBitmap(Mat image) {
        Mat temp = new Mat(image.size(), CvType.CV_8UC4);
        Imgproc.cvtColor(image, temp, Imgproc.COLOR_BGR2RGBA);
        Bitmap bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(temp, bitmap);
        return bitmap;
    }

    /**
     * Process detection results and return a map of detected items with counts
     */
    private Map<String, Integer> processDetections(List<Detection> detections) {
        Map<String, Integer> itemCounts = new HashMap<>();
        
        if (detections == null || detections.isEmpty()) {
            return itemCounts;
        }

        for (Detection detection : detections) {
            if (!detection.getCategories().isEmpty()) {
                float confidence = detection.getCategories().get(0).getScore();
                if (confidence > THRESHOLD) {
                    String label = detection.getCategories().get(0).getLabel();
                    itemCounts.put(label, itemCounts.getOrDefault(label, 0) + 1);
                }
            }
        }

        return itemCounts;
    }
}
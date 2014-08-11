/*
 * Copyright (C) 2010,2011,2012 Samuel Audet
 *
 * FacePreview - A fusion of OpenCV's facedetect and Android's CameraPreview samples,
 *               with JavaCV + JavaCPP as the glue in between.
 *
 * This file was based on CameraPreview.java that came with the Samples for 
 * Android SDK API 8, revision 1 and contained the following copyright notice:
 *
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * IMPORTANT - Make sure the AndroidManifest.xml file looks like this:
 *
 * <?xml version="1.0" encoding="utf-8"?>
 * <manifest xmlns:android="http://schemas.android.com/apk/res/android"
 *     package="com.googlecode.javacv.facepreview"
 *     android:versionCode="1"
 *     android:versionName="1.0" >
 *     <uses-sdk android:minSdkVersion="4" />
 *     <uses-permission android:name="android.permission.CAMERA" />
 *     <uses-feature android:name="android.hardware.camera" />
 *     <application android:label="@string/app_name">
 *         <activity
 *             android:name="FacePreview"
 *             android:label="@string/app_name"
 *             android:screenOrientation="landscape">
 *             <intent-filter>
 *                 <action android:name="android.intent.action.MAIN" />
 *                 <category android:name="android.intent.category.LAUNCHER" />
 *             </intent-filter>
 *         </activity>
 *     </application>
 * </manifest>
 */

package com.example.facepreview;

import android.app.Activity;



import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.IBinder;
import android.os.Vibrator;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_objdetect;
import com.googlecode.javacv.cpp.opencv_objdetect.CvHaarClassifierCascade;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;

// ----------------------------------------------------------------------

public class FacePreview extends Activity {
    private FrameLayout layout;
    private FaceView faceView;
    private Preview mPreview;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Hide the window title.
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        // Create our Preview view and set it as the content of our activity.
        try {
            layout = new FrameLayout(this);
            faceView = new FaceView(this);
            mPreview = new Preview(this, faceView);
            layout.addView(mPreview);
            layout.addView(faceView);
            setContentView(layout);
        } catch (IOException e) {
            e.printStackTrace();
            new AlertDialog.Builder(this).setMessage(e.getMessage()).create().show();
        }
    }
}
// ----------------------------------------------------------------------

class FaceView extends View implements Camera.PreviewCallback {
    public static final int SUBSAMPLING_FACTOR = 4;

    private IplImage grayImage;
    private CvHaarClassifierCascade classifier1;
    private CvHaarClassifierCascade classifier2;
    private CvHaarClassifierCascade classifier3;
    private CvMemStorage storage;
    private CvSeq faces;
    private CvSeq cars;
    private CvSeq Wallclocks;
    private Vibrator mVibrator;

    public FaceView(FacePreview context) throws IOException {
        super(context);

        // Load the classifier file from Java resources.
        //File classifierFile = Loader.extractResource(getClass(),
         //   "haarcascade_frontalface_alt.xml",
          //  context.getCacheDir(), "classifier", ".xml");
        
        File classifierFile1 = Loader.extractResource(getClass(),
                   "haarcascade_frontalface_alt.xml",
                   context.getCacheDir(), "classifier", ".xml");
        File classifierFile2 = Loader.extractResource(getClass(),
                "cars3.xml",
                context.getCacheDir(), "classifier", ".xml");
        File classifierFile3 = Loader.extractResource(getClass(),
                "classifier_WallClock.xml",
                context.getCacheDir(), "classifier", ".xml");
     
        if (classifierFile1 == null || classifierFile1.length() <= 0) {
            throw new IOException("Could not extract the classifier file1 from Java resource.");
        }
        if (classifierFile2 == null || classifierFile2.length() <= 0) {
            throw new IOException("Could not extract the classifier file2 from Java resource.");
        }
        if (classifierFile3 == null || classifierFile3.length() <= 0) {
            throw new IOException("Could not extract the classifier file3 from Java resource.");
        }

        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);
        classifier1 = new CvHaarClassifierCascade(cvLoad(classifierFile1.getAbsolutePath()));
        classifier2 = new CvHaarClassifierCascade(cvLoad(classifierFile2.getAbsolutePath()));
        classifier3 = new CvHaarClassifierCascade(cvLoad(classifierFile3.getAbsolutePath()));
        //classifier = new CvHaarClassifierCascade(classifier1);
        classifierFile1.delete();
        classifierFile2.delete();
        classifierFile3.delete();
        if (classifier1.isNull()) {
            throw new IOException("Could not load the classifier file.");
        }
        if (classifier2.isNull()) {
            throw new IOException("Could not load the classifier file.");
        }
        if (classifier3.isNull()) {
            throw new IOException("Could not load the classifier file.");
        }
        storage = CvMemStorage.create();
     // Get instance of Vibrator from current Context
        mVibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
    }

    public void onPreviewFrame(final byte[] data, final Camera camera) {
        try {
            Camera.Size size = camera.getParameters().getPreviewSize();
            processImage(data, size.width, size.height);
            camera.addCallbackBuffer(data);
        } catch (RuntimeException e) {
            // The camera has probably just been released, ignore.
        }
    }

    protected void processImage(byte[] data, int width, int height) {
        // First, downsample our image and convert it into a grayscale IplImage
        int f = SUBSAMPLING_FACTOR;
        if (grayImage == null || grayImage.width() != width/f || grayImage.height() != height/f) {
            grayImage = IplImage.create(width/f, height/f, IPL_DEPTH_8U, 1);
        }
        int imageWidth  = grayImage.width();
        int imageHeight = grayImage.height();
        int dataStride = f*width;
        int imageStride = grayImage.widthStep();
        ByteBuffer imageBuffer = grayImage.getByteBuffer();
        for (int y = 0; y < imageHeight; y++) {
            int dataLine = y*dataStride;
            int imageLine = y*imageStride;
            for (int x = 0; x < imageWidth; x++) {
                imageBuffer.put(imageLine + x, data[dataLine + f*x]);
            }
        }

        cvClearMemStorage(storage);
        faces = cvHaarDetectObjects(grayImage, classifier1, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
        //postInvalidate();
        cars = cvHaarDetectObjects(grayImage, classifier2, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
     
        Wallclocks = cvHaarDetectObjects(grayImage, classifier3, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
      
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setTextSize(20);

        String s = "Object Detector - This side up.";
        float textWidth = paint.measureText(s);
        canvas.drawText(s, (getWidth()-textWidth)/2, 20, paint);

        if (faces != null) {
            paint.setStrokeWidth(10);
            paint.setStyle(Paint.Style.STROKE);
            float scaleX = (float)getWidth()/grayImage.width();
            float scaleY = (float)getHeight()/grayImage.height();
            int total = faces.total();
            // Vibrate for 300 milliseconds
            if(total != 0)
            	  mVibrator.vibrate(100);
           //int previous = 0;
           //int sum_pixels = 0;
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                canvas.drawRect(x*scaleX, y*scaleY, (x+w)*scaleX, (y+h)*scaleY, paint);
                //sum_pixels += w*h;     
            }
           // previous = sum_pixels;
        }
        if (cars != null) {
            paint.setStrokeWidth(10);
            paint.setStyle(Paint.Style.STROKE);
            float scaleX = (float)getWidth()/grayImage.width();
            float scaleY = (float)getHeight()/grayImage.height();
            int total = cars.total();
            // Vibrate for 300 milliseconds
            if(total != 0)
            	  mVibrator.vibrate(100);
         
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(cars, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                canvas.drawRect(x*scaleX, y*scaleY, (x+w)*scaleX, (y+h)*scaleY, paint);
              
            }
        }
        if ( Wallclocks != null) {
            paint.setStrokeWidth(10);
            paint.setStyle(Paint.Style.STROKE);
            float scaleX = (float)getWidth()/grayImage.width();
            float scaleY = (float)getHeight()/grayImage.height();
            int total = Wallclocks.total();
            // Vibrate for 300 milliseconds
            if(total != 0)
            	  mVibrator.vibrate(100);
         
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(Wallclocks, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                canvas.drawRect(x*scaleX, y*scaleY, (x+w)*scaleX, (y+h)*scaleY, paint);
              
            }
        }
    }
}

// ----------------------------------------------------------------------

class Preview extends SurfaceView implements SurfaceHolder.Callback {
    SurfaceHolder mHolder;
    Camera mCamera;
    Camera.PreviewCallback previewCallback;

    Preview(Context context, Camera.PreviewCallback previewCallback) {
        super(context);
        this.previewCallback = previewCallback;

        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        // The Surface has been created, acquire the camera and tell it where
        // to draw.
        mCamera = Camera.open();
        try {
           mCamera.setPreviewDisplay(holder);
        } catch (IOException exception) {
            mCamera.release();
            mCamera = null;
            // TODO: add more exception handling logic here
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // Surface will be destroyed when we return, so stop the preview.
        // Because the CameraDevice object is not a shared resource, it's very
        // important to release it when the activity is paused.
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;
    }


    private Size getOptimalPreviewSize(List<Size> sizes, int w, int h) {
        final double ASPECT_TOLERANCE = 0.05;
        double targetRatio = (double) w / h;
        if (sizes == null) return null;

        Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        // Try to find an size match aspect ratio and size
        for (Size size : sizes) {
            double ratio = (double) size.width / size.height;
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE) continue;
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        // Cannot find the one match the aspect ratio, ignore the requirement
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }
        return optimalSize;
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        // Now that the size is known, set up the camera parameters and begin
        // the preview.
        Camera.Parameters parameters = mCamera.getParameters();

        List<Size> sizes = parameters.getSupportedPreviewSizes();
        Size optimalSize = getOptimalPreviewSize(sizes, w, h);
        parameters.setPreviewSize(optimalSize.width, optimalSize.height);

        mCamera.setParameters(parameters);
        if (previewCallback != null) {
            mCamera.setPreviewCallbackWithBuffer(previewCallback);
            Camera.Size size = parameters.getPreviewSize();
            byte[] data = new byte[size.width*size.height*
                    ImageFormat.getBitsPerPixel(parameters.getPreviewFormat())/8];
            mCamera.addCallbackBuffer(data);
        }
        mCamera.startPreview();
    }

}

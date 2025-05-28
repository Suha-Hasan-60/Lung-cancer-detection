package com.gnits.myapplication

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var predictionText: TextView
    private lateinit var selectImageBtn: Button
    private lateinit var tflite: Interpreter

    private val IMAGE_PICK_CODE = 100
    private val PERMISSION_REQUEST_CODE = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        predictionText = findViewById(R.id.predictionText)
        selectImageBtn = findViewById(R.id.selectImageBtn)

        try {
            tflite = Interpreter(loadModelFile())
        } catch (e: IOException) {
            predictionText.text = "❌ Failed to load model."
            return
        }

        if (!hasPermissions()) {
            requestPermissions()
        }

        selectImageBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.addCategory(Intent.CATEGORY_OPENABLE)
            intent.type = "image/*"
            startActivityForResult(intent, IMAGE_PICK_CODE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == IMAGE_PICK_CODE && resultCode == Activity.RESULT_OK && data != null) {
            val uri = data.data
            try {
                if (uri != null) {
                    val bitmap = getBitmapFromUri(uri)
                    imageView.setImageBitmap(bitmap)
                    val result = runModel(bitmap)
                    predictionText.text = result
                } else {
                    predictionText.text = "❌ Invalid image URI"
                }
            } catch (e: Exception) {
                Log.e("TFLiteDebug", "Error processing image", e)
                predictionText.text = "❌ Error processing image"
            }
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("vgg16_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun getBitmapFromUri(uri: Uri): Bitmap {
        return contentResolver.openInputStream(uri)?.use { inputStream ->
            val bitmap = BitmapFactory.decodeStream(inputStream)
            bitmap?.copy(Bitmap.Config.ARGB_8888, true)
                ?: throw IOException("Unable to decode image")
        } ?: throw IOException("Unable to open input stream")
    }

    private fun runModel(bitmap: Bitmap): String {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (x in 0 until 224) {
            for (y in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                input[0][x][y][0] = (pixel shr 16 and 0xFF) / 255f
                input[0][x][y][1] = (pixel shr 8 and 0xFF) / 255f
                input[0][x][y][2] = (pixel and 0xFF) / 255f
            }
        }

        val output = Array(1) { FloatArray(1) }
        tflite.run(input, output)

        return if (output[0][0] > 0.5f) "✅ Prediction: Normal" else "✅ Prediction: Lung Cancer"
    }

    private fun hasPermissions(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_MEDIA_IMAGES),
                PERMISSION_REQUEST_CODE
            )
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),
                PERMISSION_REQUEST_CODE
            )
        }
    }
}

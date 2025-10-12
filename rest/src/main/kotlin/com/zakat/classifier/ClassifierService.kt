package com.zakat.classifier

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile
import org.springframework.web.server.ResponseStatusException
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import javax.imageio.ImageIO
import kotlin.jvm.optionals.getOrNull
import kotlin.math.exp

@Service
class ClassifierService {
    companion object {
        const val MODEL_PATH = "model.onnx"
        val predictClasses = arrayOf(
            "Mild Demented",
            "Moderate Demented",
            "Non Demented",
            "Very Mild Demented",
        )
    }

    fun runModel(image: MultipartFile): PredictionResult {
        OrtEnvironment.getEnvironment().use { env ->
            env.createSession(MODEL_PATH, OrtSession.SessionOptions()).use { session ->
                val tensor = convertImageToTensor(env, image)
                val inputs = mapOf("input" to tensor)
                session.run(inputs).use { results ->
                    return generatePredictionResult(results)
                }
            }
        }
    }

    private fun generatePredictionResult(result: OrtSession.Result): PredictionResult {
        val output = result["linear_2"].getOrNull()
            ?: throw ResponseStatusException(HttpStatus.BAD_REQUEST)
        if (output !is OnnxTensor)
            throw ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Answer is not tensor")

        val value = output.value as Array<FloatArray>
        val logits = value[0]
        val probs = softmax(logits)
        val predictedClass = argmax(probs)

        return PredictionResult(
            probabilities = probs,
            predictedClass = predictedClass,
            predictedClassTitle = predictClasses[predictedClass],
        )
    }

    private fun argmax(nums: List<Float>): Int {
        var argMax = 0
        for (i in nums.indices) {
            if (nums[i] > nums[argMax]) {
                argMax = i
            }
        }

        return argMax
    }

    private fun softmax(logits: FloatArray): List<Float> {
        val expSum = logits
            .map { logit -> exp(logit) }
            .sum()

        return logits.map { logit -> exp(logit) / expSum }
    }

    private fun convertImageToTensor(env: OrtEnvironment, image: MultipartFile): OnnxTensor =
        image.inputStream.use { stream ->
            val bufferedImage = ImageIO.read(stream)
            val data = preprocessImage(bufferedImage)
            val buffer = FloatBuffer.wrap(data)

            val shape = longArrayOf(1, 3, 200, 200)
            OnnxTensor.createTensor(env, buffer, shape)
        }

    private fun preprocessImage(image: BufferedImage): FloatArray {
        val resizedImage = resizeImage(image, 200, 200)
        val width = resizedImage.width
        val height = resizedImage.height

        val data = FloatArray(3 * 200 * 200)
        val pixels = resizedImage.getRGB(0, 0, width, height, null, 0, width)

        for (i in 0 until pixels.size) {
            val pixel = pixels[i]
            var r = ((pixel shr 16) and 0xFF) / 255.0f
            var g = ((pixel shr 8) and 0xFF) / 255.0f
            var b = (pixel and 0xFF) / 255.0f

            r = (r - .5f) / .5f
            g = (r - .5f) / .5f
            b = (r - .5f) / .5f

            val row = i / width
            val col = i % width

            data[row * width + col] = r
            data[1 * height * width + row * width + col] = g
            data[2 * height * width + row * width + col] = b
        }

        return data
    }

    private fun resizeImage(image: BufferedImage, width: Int, height: Int): BufferedImage {
        val g = image.createGraphics()
        g.setRenderingHints(
            mapOf(
                RenderingHints.KEY_INTERPOLATION to RenderingHints.VALUE_INTERPOLATION_BILINEAR,
            )
        )
        g.drawImage(image, 0, 0, width, height, null)
        g.dispose()
        return image
    }
}
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
        val probs = softmax(value[0])
        val predictedClass = argmax(probs)

        return PredictionResult(
            probabilities = probs,
            predictedClass = predictedClass,
            predictedClassTitle = predictClasses[predictedClass],
        )
    }
}
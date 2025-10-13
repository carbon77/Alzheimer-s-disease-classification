package com.zakat.classifier.services

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.zakat.classifier.models.PredictionResult
import com.zakat.classifier.argmax
import com.zakat.classifier.convertImageToTensor
import com.zakat.classifier.softmax
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile
import org.springframework.web.server.ResponseStatusException
import kotlin.jvm.optionals.getOrNull

@Service
class ModelService {
    companion object {
        const val MODEL_PATH = "model.onnx"
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
        )
    }
}
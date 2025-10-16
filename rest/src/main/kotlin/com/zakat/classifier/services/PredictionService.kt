package com.zakat.classifier.services

import com.zakat.classifier.models.Prediction
import com.zakat.classifier.models.repositories.PredictionRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import org.springframework.web.server.ResponseStatusException
import java.util.*
import kotlin.jvm.optionals.getOrNull

@Service
class PredictionService(
    private val modelService: ModelService,
    private val predictionRepository: PredictionRepository,
    private val s3Service: S3Service,
) {

    @Transactional
    fun predictClass(image: MultipartFile, save: Boolean): Prediction {
        val result = modelService.runModel(image)
        val s3Key = if (save) s3Service.putMultipartFile(image) else ""
        var prediction = Prediction(
            s3Key = s3Key,
            logits = result.logits,
            probabilities = result.probabilities,
            predictedClass = result.predictedClass,
        )

        if (save) {
            prediction = predictionRepository.save(prediction)
        }
        return prediction
    }

    @Transactional
    fun deletePrediction(predictionId: UUID) {
        val prediction = predictionRepository.findById(predictionId).getOrNull() ?: return

        if (prediction.s3Key.isNotEmpty()) {
            s3Service.deleteObject(prediction.s3Key)
        }

        predictionRepository.delete(prediction)
    }

    @Transactional(readOnly = true)
    fun downloadImage(predictionId: UUID): DownloadImageResult {
        val prediction = predictionRepository.findById(predictionId).getOrNull()
        if (prediction == null || prediction.s3Key.isEmpty()) {
            throw ResponseStatusException(HttpStatus.NOT_FOUND)
        }

        val fileData = s3Service.getObject(prediction.s3Key)
        return DownloadImageResult(
            data = fileData,
            filename = prediction.s3Key.split("__").last(),
        )
    }

    fun findAll(): Iterable<Prediction> {
        return predictionRepository.findAll()
    }
}

data class DownloadImageResult(
    val data: ByteArray,
    val filename: String,
)
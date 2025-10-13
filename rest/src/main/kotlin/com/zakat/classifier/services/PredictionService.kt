package com.zakat.classifier.services

import com.zakat.classifier.models.Prediction
import com.zakat.classifier.models.repositories.PredictionRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.multipart.MultipartFile
import java.util.*
import kotlin.jvm.optionals.getOrNull

@Service
class PredictionService(
    private val modelService: ModelService,
    private val predictionRepository: PredictionRepository,
    private val s3Service: S3Service,
) {

    @Transactional
    fun predictClass(image: MultipartFile): Prediction {
        val result = modelService.runModel(image)
        val s3Key = s3Service.putMultipartFile(image)
        var prediction = Prediction(
            s3Key = s3Key,
            logits = result.logits,
            probabilities = result.probabilities,
            predictedClass = result.predictedClass,
        )
        prediction = predictionRepository.save(prediction)
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

    fun findAll(): Iterable<Prediction> {
        return predictionRepository.findAll()
    }
}
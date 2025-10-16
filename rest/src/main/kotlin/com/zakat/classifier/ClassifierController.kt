package com.zakat.classifier

import com.zakat.classifier.models.Prediction
import com.zakat.classifier.services.PredictionService
import io.swagger.v3.oas.annotations.Operation
import org.springframework.core.io.ByteArrayResource
import org.springframework.core.io.Resource
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile
import java.util.*

@RestController
@RequestMapping("/predict")
class ClassifierController(
    private val predictionService: PredictionService,
) {

    @Operation(summary = "Predict class from the image")
    @PostMapping
    fun runModel(
        @RequestParam("image") image: MultipartFile,
        @RequestParam("save") save: Boolean,
    ): Prediction {
        return predictionService.predictClass(image, save)
    }

    @Operation(summary = "Get all predictions")
    @GetMapping("all")
    fun findAll(): Iterable<Prediction> = predictionService.findAll()

    @Operation(summary = "Delete prediction")
    @DeleteMapping("{predictionId}")
    fun deletePrediction(@PathVariable("predictionId") predictionId: UUID) {
        predictionService.deletePrediction(predictionId)
    }

    @Operation(summary = "Download image by prediction id")
    @GetMapping("{predictionId}/download")
    fun downloadPrediction(@PathVariable("predictionId") predictionId: UUID): ResponseEntity<Resource> {
        val (data, filename) = predictionService.downloadImage(predictionId)
        val resource = ByteArrayResource(data)

        return ResponseEntity.ok()
            .contentType(MediaType.APPLICATION_OCTET_STREAM)
            .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"$filename\"")
            .body(resource)
    }
}
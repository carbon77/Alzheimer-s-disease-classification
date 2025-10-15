package com.zakat.classifier.services

import com.zakat.classifier.config.S3Properties
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import java.util.*

@Service
class S3Service(
    private val s3Properties: S3Properties,
    private val s3Client: S3Client,
) {

    fun putMultipartFile(file: MultipartFile): String {
        val filename = generateName(file)
        val request = PutObjectRequest.builder()
            .bucket(s3Properties.bucket)
            .key(filename)
            .contentType(file.contentType)
            .ifNoneMatch("*")
            .build()
        s3Client.putObject(request, RequestBody.fromBytes(file.bytes))
        return filename
    }

    fun deleteObject(key: String) {
        val request = DeleteObjectRequest.builder()
            .bucket(s3Properties.bucket)
            .key(key)
            .build()
        s3Client.deleteObject(request)
    }

    fun getObject(key: String): ByteArray {
        val request = GetObjectRequest.builder()
            .bucket(s3Properties.bucket)
            .key(key)
            .build()
        return s3Client.getObjectAsBytes(request).asByteArray()
    }

    private fun generateName(file: MultipartFile): String {
        val uuid = UUID.randomUUID()
        return "${uuid}__${file.originalFilename}"
    }
}
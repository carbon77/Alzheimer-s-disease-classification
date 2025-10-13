package com.zakat.classifier.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.context.annotation.Configuration

@Configuration
@ConfigurationProperties(prefix = "s3")
data class S3Properties(
    var region: String = "",
    var endpointUrl: String = "",
    var bucket: String = "",
    var accessKey: String = "",
    var secretKey: String = "",
)
package com.zakat.classifier.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.AwsCredentials
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.s3.S3Client
import java.net.URI

@Configuration
class S3Config {
    @Bean
    fun awsCredentials(
        s3Properties: S3Properties,
    ): AwsBasicCredentials {
        return AwsBasicCredentials.create(s3Properties.accessKey, s3Properties.secretKey)
    }

    @Bean
    fun s3Client(
        s3Properties: S3Properties,
        awsCredentials: AwsCredentials,
    ): S3Client {
        return S3Client
            .builder()
            .region(Region.of(s3Properties.region))
            .credentialsProvider(StaticCredentialsProvider.create(awsCredentials))
            .endpointOverride(URI.create(s3Properties.endpointUrl))
            .build()
    }
}
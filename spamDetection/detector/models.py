from django.db import models

class Message(models.Model):
    text = models.TextField(help_text="Enter the message text.")
    is_spam = models.BooleanField(default=False, help_text="Indicates if the message is spam.")
    prediction_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score of the spam prediction.")
    created_at = models.DateTimeField(auto_now_add=True, help_text="Timestamp when the message was created.")

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Message"
        verbose_name_plural = "Messages"

    def __str__(self):
        return f"{'Spam' if self.is_spam else 'Ham'}: {self.text[:50]}... (Confidence: {self.prediction_confidence})"

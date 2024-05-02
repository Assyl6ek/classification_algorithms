from django.db import models
# Create your models here.


class comment(models.Model):
    id = models.BigIntegerField(primary_key=True)
    restaurant_id = models.BigIntegerField()
    title = models.CharField(max_length=255)
    text = models.CharField(max_length=255)
    user_id = models.BigIntegerField()
    author_name = models.CharField(max_length=255)
    author_avatar = models.CharField(max_length=255)
    publication_date = models.DateField()
    is_positive = models.IntegerField()
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

class Review(models.Model):
    id = models.BigIntegerField(primary_key=True)
    is_positive = models.IntegerField()
    text = models.CharField(max_length=255)
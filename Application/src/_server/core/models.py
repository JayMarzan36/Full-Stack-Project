from django.db import models
from django.contrib.auth.models import User


class Note(models.Model):
    # Notes need
    # User
    # ID, gets from database
    # Content
    # Relation with other docs
    # - Next Closest doc

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # ID for notes comes from the database
    content = models.CharField()
    biggestConcept = models.CharField()

    # Many to many because one note might be the closest to another, but that other might be closer to a different one
    nextClosestDoc = models.ManyToManyField("Note")

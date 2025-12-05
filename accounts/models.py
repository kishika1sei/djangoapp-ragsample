from django.db import models
from django.contrib.auth.models import AbstractUser

class Department(models.Model):
    name = models.CharField("部門名", max_length=100, unique=True)

    class Meta:
        verbose_name = "部門"
        verbose_name_plural = "部門"

    def __str__(self) -> str:
        return self.name
    
class User(AbstractUser):
    class Role(models.TextChoices):
        SYSTEM_ADMIN = "system_admin", "システム管理者"
        DEPT_ADMIN = "dept_admin", "部門管理者"
        DEPT_STAFF = "dept_staff", "部門ユーザー"

    department = models.ForeignKey(
        Department,
        verbose_name="部門",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="users",
    )
    role = models.CharField(
        "権限",
        max_length=20,
        choices=Role.choices,
        default=Role.DEPT_STAFF,
    )

    class Meta:
        verbose_name = "ユーザー"
        verbose_name_plural = "ユーザー"
    
    def __str__(self) -> str:
        return self.username
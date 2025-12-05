from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from .models import User, Department
# Register your models here.

@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ("id","name")
    search_fields = ("name",)

@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    # 一覧に表示する項目
    list_display = (
        "id",
        "username",
        "email",
        "department",
        "role",
        "is_active",
        "is_staff",
        "is_superuser",
    )
    list_filter = ("role","department","is_active","is_staff","is_superuser")
    search_fields = ("username","email")

    # 編集画面のレイアウト
    fieldsets = DjangoUserAdmin.fieldsets + (
        ("業務情報", {"fields": ("department", "role")}),
    )
    add_fieldsets = DjangoUserAdmin.add_fieldsets +(
        ("業務情報", {"fields": ("department", "role")}),
    )
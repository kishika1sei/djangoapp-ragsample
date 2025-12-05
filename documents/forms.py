from django import forms
from .models import Document

class DocumentUploadForm(forms.Form):
    file = forms.FileField(label='ファイル')
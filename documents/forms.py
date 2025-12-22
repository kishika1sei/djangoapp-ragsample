from django import forms
from accounts.models import Department


class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    """
    Django標準の FileField は「単一ファイル」前提。
    multiple で来たリストを受けられるように拡張する。
    """
    def clean(self, data, initial=None):
        if data is None:
            return []

        if isinstance(data, (list, tuple)):
            cleaned = []
            for item in data:
                cleaned.append(super().clean(item, initial))
            return cleaned

        return [super().clean(data, initial)]


class DocumentUploadForm(forms.Form):
    files = MultipleFileField(
        label="ファイル（複数選択可）",
        widget=MultiFileInput(attrs={"multiple": True}),
        required=True,
    )

    department = forms.ModelChoiceField(
        label="部門（superuserのみ選択可）",
        queryset=Department.objects.all(),
        required=False,
        empty_label="選択してください",
    )

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not (user and user.is_superuser):
            self.fields.pop("department", None)

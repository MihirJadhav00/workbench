
from django.contrib import admin
from django.urls import path
from .views import HomePage,ConfirmPage

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',HomePage),
    path('confirmation/',ConfirmPage)
]

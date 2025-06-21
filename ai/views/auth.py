from restframework.views import APIView
from restframework.generic.viewset import ModelView
from restframework.response import Response


class AuthAPIView(APIView):
    queryset = User.objects.filter()
    serializer_class = Serializer() 

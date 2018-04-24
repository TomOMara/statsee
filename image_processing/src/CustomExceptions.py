class NoCurvesFoundException(Exception):
     def __init__(self, image_name):
         self.image_name = image_name
     def __str__(self):
         return repr("No curves found for image " + self.image_name)

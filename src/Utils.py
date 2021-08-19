class Utils():
	def readFiles():
		return None,None

	def readAnnotations():
		return []

	def readMesh(meshFileLocation):
		import json
		f = open(meshFileLocation,)
		data = json.load(f)
		f.close()
		return data
		
	def buildIndentificationSubmission(indentifiedChemicals):
		print("Write file")
		
	def buildIndexingSubmission(indexedChemicals):
		print("Write file")


# From Rui Antunes utils.py
import logging
import os
import sys
log = logging.getLogger(__name__)
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


LOG_FILE = os.path.join("logs", "logsbiocreative.log")

class BaseLogger:
	def __init__(self):
		"""
        From: https://www.toptal.com/python/in-depth-python-logging
        """
		super().__init__()

		self.logger = logging.getLogger(self.__class__.__name__)

		if not self.logger.hasHandlers():
			self.logger.setLevel(logging.DEBUG)

			console_handler = logging.StreamHandler(sys.stdout)
			console_handler.setFormatter(FORMATTER)
			self.logger.addHandler(console_handler)

			if not os.path.exists('logs'):
				os.makedirs('logs')

			file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding='utf-8')
			file_handler.setFormatter(FORMATTER)
			self.logger.addHandler(file_handler)

			self.logger.propagate = False
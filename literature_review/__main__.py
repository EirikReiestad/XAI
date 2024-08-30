from literature_review.src.handler import Handler
from literature_review import settings
import os

writer_filename = os.path.join(settings.BASE_DIR, settings.MAIN_FILE)
review_filename = os.path.join(settings.BASE_DIR, settings.REVIEW_FILE)

handler = Handler(writer_filename)
handler.run(review_filename)

from literature_review import settings
import os


assert os.path.exists(settings.BASE_DIR), f"{settings.BASE_DIR} does not exist"
main_file = os.path.join(settings.BASE_DIR, settings.MAIN_FILE)
review_file = os.path.join(settings.BASE_DIR, settings.REVIEW_FILE)
assert os.path.exists(main_file), f"{main_file} does not exist"
assert os.path.exists(review_file), f"{review_file} does not exist"

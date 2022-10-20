import hashlib
import logging
import os

logger = logging.getLogger(__name__)

################################################################################
################################################################################


def compare_folders(expected, actual):
    """Compare the contents in the folder for differences.

    Only supports files and subfolders at level 1 (i.e. no subfolder of subfolder).

    :param expected: Folder with expected files.
    :param actual: Folder with actual files, i.e. output from tests.
    :return: None.
    """
    compare_files_in_folders(expected, actual)

    checked_folders = set()
    for path_ in os.listdir(expected):
        path_e = os.path.join(expected, path_)
        path_a = os.path.join(actual, path_)
        checked_folders.add(path_e)
        checked_folders.add(path_a)
        if os.path.isdir(path_e):
            compare_files_in_folders(path_e, path_a)
    for path_ in os.listdir(actual):
        path_a = os.path.join(actual, path_)
        if path_a not in checked_folders:
            logger.error("Expected missing folder: %s", path_)


def file_content_checksum(filename):
    """Helper function to calculate the md5 hash of the file content.

    :param filename: File to compute the md5 hash of the content.
    :type filename: str

    :return: md5 hash
    :rtype: str
    """
    with open(filename, 'r') as f:
        contents = f.read()
        md5sum = hashlib.md5(contents.encode('utf-8')).hexdigest()
    return md5sum


def compare_files_in_folders(expected, actual):
    """Compares the contents of the folders.

    Currently only checks CSV files.
    Raises assertion error if files are different or missing.

    :param expected: Check the expected folder against the actual folder.
    :param actual:
    """
    mismatch = False
    checked_files = set()

    for f1 in os.listdir(expected):
        if f1.endswith('.csv'):
            fp1 = os.path.join(expected, f1)
            fp2 = os.path.join(actual, f1)
            logger.info('Comparing: %s %s', fp1, fp2)

            if not os.path.isfile(fp2):
                logger.error("Actual missing file: %s", f1)
                mismatch = True
            else:
                if file_content_checksum(fp1) != file_content_checksum(fp2):
                    logger.error("File contents differ: %s %s", fp1, fp2)
                    mismatch = True
                    checked_files.add(f1)

    for f2 in os.listdir(actual):
        if f2.endswith('.csv'):
            fp1 = os.path.join(expected, f2)
            fp2 = os.path.join(actual, f2)

            if not os.path.isfile(fp1):
                logger.error("Expected missing file: %s", f2)
                mismatch = True
            else:
                if f2 not in checked_files and file_content_checksum(fp1) != file_content_checksum(fp2):
                    logger.error("File contents differ: %s %s", fp1, fp2)
                    mismatch = True

    if mismatch:
        raise AssertionError("Mismatching folder contents.")
    else:
        logger.info("All files are the same in folders: %s %s", expected, actual)


def get_test_folders(test_folder, test_name):
    """Helper function to return the standard folder structure of the test data

    :param test_folder: Folder containing the test data
    :type test_folder: str
    :param test_name: Name of the test
    :type test_name: str

    :return: folders of the standard test structure
    :rtype: str, str, str
    """
    expected_folder = os.path.join(*[test_folder, test_name, 'expected'])
    input_folder = os.path.join(*[test_folder, test_name, 'input'])
    output_folder = os.path.join(*[test_folder, test_name, 'output'])

    os.makedirs(output_folder, exist_ok=True)

    return expected_folder, input_folder, output_folder


# CoNLLU Toolkit: Generic Meta Reader for CoNLL formats
#
# Author: Ishan Jindal <ishan.jindal@ibm.com>
#


import json
import re


class MetaInfo():
    """Represents the metadata.
    """
    def __init__(self, meta):
        self.meta = meta
        self.sen_id = ""
        self.sen_txt = ""
        self.srl = ""
        self.process_meta()

    def process_meta(self):
        for meta_line in self.meta:
            if self.sen_txt == "":
                if re.match(r'^# [0-9]+ #', meta_line):#len(meta_line.split("#")) >= 3: #
                    self.sen_id = meta_line.split("#")[1] # Sentence id is not necessarily integer
                    self.sen_txt = "#".join(meta_line.split("#")[2:]).strip()
                elif meta_line.startswith("sentence-text:"):
                    self.sen_txt = "sentence-text:".join(meta_line.split("sentence-text:")[1:]).strip()
                    continue
                elif meta_line.startswith("# text = "):
                    self.sen_txt = "text = ".join(meta_line.split("text = ")[1:]).strip()
                    continue
                elif len(meta_line.split("sentence-text:")) > 1 and not re.match(r'^# \d # ', meta_line):
                    self.sen_txt = "#".join(meta_line.split("#")[1:]).strip()

            if meta_line.startswith("# srl = "):
                self.srl = json.loads("srl = ".join(meta_line.split("srl = ")[1:]).strip())
            elif meta_line.startswith("# span_srl = "):
                self.srl = json.loads("span_srl = ".join(meta_line.split("span_srl = ")[1:]).strip())
            if meta_line.startswith("# sent_id = "):
                self.sen_id = "sent_id = ".join(meta_line.split("sent_id = ")[1:]).strip()

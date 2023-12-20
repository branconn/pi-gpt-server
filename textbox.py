import tkinter as tk

class AutocompleteText(tk.Text):
    def __init__(self, *args, **kwargs):
        self.callback = kwargs.pop("autocomplete", None)
        super().__init__(*args, **kwargs)

        # bind on key release, which will happen after tkinter
        # inserts the typed character
        self.bind("<Any-KeyRelease>", self._autocomplete)

        # special handling for tab, which needs to happen on the
        # key _press_
        self.bind("<Tab>", self._handle_tab)

    def _handle_tab(self, event):
        # see if any text has the "autocomplete" tag
        tag_ranges= self.tag_ranges("autocomplete")
        if tag_ranges:
            # move the insertion cursor to the end of
            # the selected text, and then remove the "sel"
            # and "autocomplete" tags
            self.mark_set("insert", tag_ranges[1])
            self.tag_remove("sel", "1.0", "end")
            self.tag_remove("autocomplete", "1.0", "end")

            # prevent the default behavior of inserting a literal tab
            return "break"

    def _autocomplete(self, event):
        if event.char and self.callback:
            # get word preceeding the insertion cursor
            word = self.get("insert-1c wordstart", "insert-1c wordend")

            # pass word to callback to get possible matches
            matches = self.callback(word)

            if matches:
                # autocomplete on the first match
                remainder = matches[0][len(word):]

                # remember the current insertion cursor
                insert = self.index("insert")

                # insert at the insertion cursor the remainder of
                # the matched word, and apply the tag "sel" so that
                # it is selected. Also, add the "autocomplete" text
                # which will make it easier to find later.
                self.insert(insert, remainder, ("sel", "autocomplete"))

                # move the cursor back to the saved position
                self.mark_set("insert", insert)


def get_matches(word):
    # For illustrative purposes, pull possible matches from 
    # what has already been typed. You could just as easily 
    # return a list of pre-defined keywords.
    words = text.get("1.0", "end-1c").split()
    matches = [x for x in words if x.startswith(word)]
    return matches

root = tk.Tk()
text = AutocompleteText(root, autocomplete=get_matches)
text.pack(fill="both", expand=True)

root.mainloop()
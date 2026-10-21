"""Read-only public rewind timeline and exact checkpoint selection for Tk.

The backend owns checkpoint creation, privacy filtering, session validation,
and restoration. This view never reads engine snapshots or client logs.
"""
import json
import os
import time
import tkinter as tk
from tkinter import ttk

TIMELINE_NAME = 'ai_rewind_timeline.json'
SCHEMA = 'rewind_timeline_v1'


class TimelineUnavailable(ValueError):
    pass


def read_json(path):
    try:
        with open(path, encoding='utf-8') as stream:
            value = json.load(stream)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class TimelineModel:
    """Only the documented public projection is retained for presentation."""
    def __init__(self, document):
        if document.get('schema') != SCHEMA:
            raise TimelineUnavailable('Waiting for a supported rewind timeline.')
        self.session = document.get('session')
        if not isinstance(self.session, str) or not self.session:
            raise TimelineUnavailable('The timeline has no current session.')
        raw_entries = document.get('entries')
        if not isinstance(raw_entries, list):
            raise TimelineUnavailable('The timeline is incomplete. Waiting for an update.')
        self.entries = []
        self.checkpoints = {}
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise TimelineUnavailable('The timeline contains an invalid entry.')
            checkpoint = raw.get('checkpoint_id')
            if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
                raise TimelineUnavailable('The timeline contains an invalid checkpoint ID.')
            if checkpoint in self.checkpoints:
                raise TimelineUnavailable('The timeline repeats a checkpoint ID.')
            events = raw.get('events', [])
            if not isinstance(events, list):
                raise TimelineUnavailable('The timeline contains an invalid event group.')
            # No repr(raw), internal decision params, card IDs, or hidden hand
            # fallback. The producer explicitly supplies rendered public text.
            public = []
            for event in events:
                if not isinstance(event, dict) or not isinstance(event.get('public_text'), str):
                    raise TimelineUnavailable('A timeline event has no public description.')
                public.append(event['public_text'])
            label = raw.get('label')
            if not isinstance(label, str):
                raise TimelineUnavailable('A timeline entry has no public label.')
            if raw.get('turn') is not None and type(raw['turn']) is not int:
                raise TimelineUnavailable('A timeline entry has an invalid turn.')
            entry = dict(checkpoint_id=checkpoint, turn=raw.get('turn'), label=label,
                         restorable=raw.get('restorable') is True and checkpoint is not None,
                         events=tuple(public), number=len(self.checkpoints) + 1 if checkpoint else None)
            self.entries.append(entry)
            if checkpoint is not None:
                self.checkpoints[checkpoint] = entry
        self.turn = document.get('turn')
        self.current = document.get('current_checkpoint_id')
        self.turn_start = document.get('turn_start_checkpoint_id')
        self.restored = document.get('restored_checkpoint_id')
        if self.turn is not None and type(self.turn) is not int:
            raise TimelineUnavailable('The timeline has an invalid current turn.')
        if any(value is not None and (not isinstance(value, str) or not value)
               for value in (self.current, self.turn_start, self.restored)):
            raise TimelineUnavailable('The timeline has an invalid checkpoint reference.')
        self.note = ''

    def checkpoint(self, checkpoint_id):
        entry = self.checkpoints.get(checkpoint_id)
        if not entry or not entry['restorable']:
            raise TimelineUnavailable('That exact checkpoint is no longer available. Refresh the timeline.')
        return entry

    def request(self, checkpoint_id, expected_session, epoch):
        if expected_session != self.session:
            raise TimelineUnavailable('This selection belongs to an earlier session. Select a checkpoint in the current timeline.')
        self.checkpoint(self.turn_start if checkpoint_id is None else checkpoint_id)
        return dict(rewind_turn=epoch, rewind_session=self.session,
                    rewind_checkpoint=checkpoint_id)

    def current_text(self):
        return 'Current state: turn %s | %s\nLast restored: %s' % (
            self.turn if self.turn is not None else '?',
            self.checkpoint_label(self.current) if self.current else 'between checkpoints',
            self.checkpoint_label(self.restored, detailed=True) if self.restored else 'none')

    @staticmethod
    def phase_label(entry):
        label = entry['label'].replace('_', ' ').strip()
        return label[:1].upper() + label[1:]

    def checkpoint_label(self, checkpoint_id, *, detailed=False):
        entry = self.checkpoints.get(checkpoint_id)
        if entry is None:
            return 'Unavailable checkpoint'
        label = 'Checkpoint %d' % entry['number']
        if detailed:
            label += ' | turn %s | %s' % (entry['turn'] if entry['turn'] is not None else '?',
                                          self.phase_label(entry))
        return label

    def signature(self):
        return (self.session, self.turn, self.current, self.turn_start, self.restored, self.note,
                tuple((e['checkpoint_id'], e['turn'], e['label'], e['restorable'], e['events'])
                      for e in self.entries))


def load_timeline(client_dir, status, *, now=None, boot_ts=0):
    now = time.time() if now is None else now
    try:
        stamp = float(status.get('ts', 0))
    except (TypeError, ValueError):
        stamp = 0
    if stamp < boot_ts or not 0 <= now - stamp < 12:
        raise TimelineUnavailable('Waiting for the current AI server before enabling rewind.')
    state = status.get('turn_rewind')
    if not isinstance(state, dict) or not state.get('supported') or not state.get('session'):
        raise TimelineUnavailable('Rewind is not available for this client session yet.')
    model = TimelineModel(read_json(os.path.join(client_dir, TIMELINE_NAME)))
    if model.session != state['session']:
        raise TimelineUnavailable('Waiting for the timeline for the current session.')
    model.note = state.get('note') if isinstance(state.get('note'), str) else ''
    return model


class TimelineWindow:
    """Single-click selection is read-only; the Restore button submits it."""
    def __init__(self, parent, load, restore):
        self.load = load
        self.restore = restore
        self.model = None
        self.selected = None
        self.rows = {}
        self.window = tk.Toplevel(parent)
        self.window.title('Rewind - game timeline')
        self.window.geometry('900x620')
        self.window.minsize(620, 420)
        self.window.transient(parent)
        self.current = tk.Label(self.window, anchor='w', justify=tk.LEFT,
                                font=('Segoe UI', 10, 'bold'), wraplength=860)
        self.current.pack(fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(self.window, text='Select a checkpoint, then click Restore selected. '
                 'It restores the state after all events in its group. Unfinished event groups have no restore point.',
                 anchor='w', justify=tk.LEFT, wraplength=860).pack(fill=tk.X, padx=12, pady=(0, 6))
        frame = tk.Frame(self.window)
        frame.pack(fill=tk.BOTH, expand=True, padx=12)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(frame, columns=('turn', 'label'), selectmode='browse')
        self.tree.heading('#0', text='Checkpoint / public event')
        self.tree.heading('turn', text='Turn')
        self.tree.heading('label', text='Moment')
        self.tree.column('#0', width=390, minwidth=220, stretch=True)
        self.tree.column('turn', width=60, minwidth=50, stretch=False)
        self.tree.column('label', width=400, minwidth=180, stretch=True)
        vertical = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        horizontal = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.tree.tag_configure('current', foreground='#136b36')
        self.tree.tag_configure('unavailable', foreground='#777777')
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        self.selection_label = tk.Label(self.window, text='Selected moment: none',
            anchor='w', justify=tk.LEFT, wraplength=860, font=('Segoe UI', 10, 'bold'))
        self.selection_label.pack(fill=tk.X, padx=12, pady=(8, 4))
        detail_frame = tk.Frame(self.window)
        detail_frame.pack(fill=tk.X, padx=12)
        self.detail = tk.Text(detail_frame, height=5, wrap=tk.WORD, state=tk.DISABLED,
                              font=('Segoe UI', 9))
        detail_scroll = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail.yview)
        self.detail.configure(yscrollcommand=detail_scroll.set)
        self.detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.feedback = tk.Label(self.window, text='', anchor='w', justify=tk.LEFT,
                                 wraplength=860, fg='#8a4b00')
        self.feedback.pack(fill=tk.X, padx=12, pady=4)
        buttons = tk.Frame(self.window)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 10))
        self.restore_button = tk.Button(buttons, text='Restore selected checkpoint',
            command=self.restore_selected, state=tk.DISABLED, font=('Segoe UI', 10, 'bold'))
        self.restore_button.pack(side=tk.LEFT)
        tk.Button(buttons, text='Refresh', command=self.refresh).pack(side=tk.LEFT, padx=6)
        tk.Button(buttons, text='Close', command=self.window.destroy).pack(side=tk.RIGHT)
        self.refresh()

    def exists(self):
        return bool(self.window.winfo_exists())

    def show(self):
        self.window.deiconify()
        self.window.lift()
        self.refresh()

    def set_detail(self, text):
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete('1.0', tk.END)
        self.detail.insert('1.0', text)
        self.detail.configure(state=tk.DISABLED)

    def refresh(self):
        try:
            model = self.load()
        except TimelineUnavailable as error:
            self.model = None
            self.rows = {}
            self.tree.delete(*self.tree.get_children())
            self.selected = None
            self.restore_button.configure(state=tk.DISABLED)
            self.current.configure(text='Current state unavailable')
            self.selection_label.configure(text='Selected moment: none')
            self.set_detail('')
            self.feedback.configure(text=str(error))
            return
        if self.model is not None and self.model.signature() == model.signature():
            return
        previous = self.selected
        old_view = self.tree.yview()
        if self.model and self.model.session != model.session:
            previous = None
        self.model = model
        self.rows = {}
        self.tree.delete(*self.tree.get_children())
        target = None
        for number, entry in enumerate(model.entries):
            checkpoint = entry['checkpoint_id']
            node = 'checkpoint_%d' % number
            flags = []
            if checkpoint and checkpoint == model.current:
                flags.append('CURRENT')
            if checkpoint and checkpoint == model.restored:
                flags.append('LAST RESTORED')
            text = model.checkpoint_label(checkpoint) if checkpoint else 'Public events - no restore point'
            if flags:
                text += ' [' + ', '.join(flags) + ']'
            tags = ('current',) if checkpoint and checkpoint == model.current else ()
            if not entry['restorable']:
                tags = ('unavailable',)
            self.tree.insert('', tk.END, iid=node, text=text,
                values=(entry['turn'] if entry['turn'] is not None else '?', model.phase_label(entry)),
                tags=tags, open=True)
            self.rows[node] = (entry, None)
            if previous == (model.session, checkpoint) and entry['restorable']:
                target = node
            for index, event in enumerate(entry['events']):
                child = '%s_event_%d' % (node, index)
                self.tree.insert(node, tk.END, iid=child, text=event.replace('\n', ' | '), values=('', 'Public event'))
                self.rows[child] = (entry, event)
        self.current.configure(text=model.current_text())
        self.feedback.configure(text=model.note or ('' if model.entries else 'No checkpoint has been recorded yet.'))
        self.selected = None
        self.restore_button.configure(state=tk.DISABLED)
        if target:
            self.tree.selection_set(target)
            self.on_select()
        else:
            self.selection_label.configure(text='Selected moment: none')
            self.set_detail('')
        if old_view:
            self.tree.yview_moveto(old_view[0])

    def on_select(self, _event=None):
        selected = self.tree.selection()
        self.selected = None
        self.restore_button.configure(state=tk.DISABLED)
        if not selected or selected[0] not in self.rows:
            return
        entry, event = self.rows[selected[0]]
        checkpoint = entry['checkpoint_id']
        if event is not None:
            self.selection_label.configure(text='Selected public event | turn %s | group: %s' % (
                entry['turn'], self.model.checkpoint_label(checkpoint) if checkpoint else 'no checkpoint'))
            self.set_detail(event + '\n\nSelect the checkpoint row to restore its exact saved state.')
            return
        self.selection_label.configure(text='Selected moment: %s' % (
            self.model.checkpoint_label(checkpoint, detailed=True) if checkpoint
            else 'Public events | turn %s | %s' % (entry['turn'], self.model.phase_label(entry))))
        self.set_detail('Restore state after these events:\n\n' +
                        ('\n'.join(entry['events']) or 'No public events in this checkpoint group.'))
        if entry['restorable']:
            self.selected = (self.model.session, checkpoint)
            self.restore_button.configure(state=tk.NORMAL)

    def restore_selected(self):
        if self.selected is None:
            self.feedback.configure(text='Select an available checkpoint row first.')
            return
        session, checkpoint = self.selected
        try:
            # Reload immediately before issuing a request. The server validates
            # the same session and exact ID again when it consumes the control.
            current = self.load()
            current.request(checkpoint, session, 0)
            self.restore(checkpoint, session)
        except (TimelineUnavailable, OSError) as error:
            self.restore_button.configure(state=tk.DISABLED)
            self.feedback.configure(text=str(error))
            return
        self.restore_button.configure(state=tk.DISABLED)
        self.feedback.configure(text='Restore requested: %s. Waiting for the client and AI to confirm.' %
                                current.checkpoint_label(checkpoint, detailed=True))

from pyscipopt import Eventhdlr, SCIP_EVENTTYPE


class DebugEvents(Eventhdlr):
    def __init__(self, debug_events=['LPSOLVED','ROWADDEDSEPA','ROWADDEDLP','NODEBRANCHED','UBTIGHTENED','ROWDELETEDLP','ROWDELETEDSEPA']):
        self.debug_LPSOLVED = 'LPSOLVED' in debug_events
        self.debug_ROWADDEDSEPA = 'ROWADDEDSEPA' in debug_events
        self.debug_ROWADDEDLP = 'ROWADDEDLP' in debug_events
        self.debug_NODEBRANCHED = 'NODEBRANCHED' in debug_events
        self.debug_UBTIGHTENED = 'UBTIGHTENED' in debug_events
        self.debug_ROWDELETEDLP = 'ROWDELETEDLP' in debug_events
        self.debug_ROWDELETEDSEPA = 'ROWDELETEDSEPA' in debug_events

    def eventinit(self):
        print('LPSOLVED eventinit')
        if self.debug_LPSOLVED:
            self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)
        if self.debug_ROWADDEDSEPA:
            self.model.catchEvent(SCIP_EVENTTYPE.ROWADDEDSEPA, self)
        if self.debug_ROWADDEDLP:
            self.model.catchEvent(SCIP_EVENTTYPE.ROWADDEDLP, self)
        if self.debug_NODEBRANCHED:
            self.model.catchEvent(SCIP_EVENTTYPE.NODEBRANCHED, self)
        if self.debug_UBTIGHTENED:
            self.model.catchEvent(SCIP_EVENTTYPE.UBTIGHTENED, self)
        if self.debug_ROWDELETEDLP:
            self.model.catchEvent(SCIP_EVENTTYPE.ROWDELETEDLP, self)
        if self.debug_ROWDELETEDSEPA:
            self.model.catchEvent(SCIP_EVENTTYPE.ROWDELETEDSEPA, self)

    def eventexit(self):
        print('LPSOLVED eventexit')
        if self.debug_LPSOLVED:
            self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)
        if self.debug_ROWADDEDSEPA:
            self.model.dropEvent(SCIP_EVENTTYPE.ROWADDEDSEPA, self)
        if self.debug_ROWADDEDLP:
            self.model.dropEvent(SCIP_EVENTTYPE.ROWADDEDLP, self)
        if self.debug_NODEBRANCHED:
            self.model.dropEvent(SCIP_EVENTTYPE.NODEBRANCHED, self)
        if self.debug_UBTIGHTENED:
            self.model.dropEvent(SCIP_EVENTTYPE.UBTIGHTENED, self)
        if self.debug_ROWDELETEDLP:
            self.model.dropEvent(SCIP_EVENTTYPE.ROWDELETEDLP, self)
        if self.debug_ROWDELETEDSEPA:
            self.model.dropEvent(SCIP_EVENTTYPE.ROWDELETEDSEPA, self)

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.LPSOLVED:
            print('event - LPSOLVED')
        elif event.getType() == SCIP_EVENTTYPE.ROWADDEDSEPA:
            print('event - ROWADDEDSEPA')
        elif event.getType() == SCIP_EVENTTYPE.ROWADDEDLP:
            print('event - ROWADDEDLP')
        elif event.getType() == SCIP_EVENTTYPE.NODEBRANCHED:
            print('event - NODEBRANCHED')
        elif event.getType() == SCIP_EVENTTYPE.UBTIGHTENED:
            print('event - UBTIGHTENED')
        elif event.getType() == SCIP_EVENTTYPE.ROWDELETEDSEPA:
            print('event - ROWDELETEDSEPA')
        elif event.getType() == SCIP_EVENTTYPE.ROWDELETEDLP:
            print('event - ROWDELETEDLP')
        else:
            raise ValueError('event error')


class BranchingEventHdlr(Eventhdlr):
    # detect branching
    def __init__(self, on_nodebranched_event, on_lpsolved_event):
        """
        execute callbacks on events
        :param on_nodebranched_event: callable
        :param on_lpsolved_event: callable
        """
        self.on_nodebranched_event = on_nodebranched_event
        self.on_lpsolved_event = on_lpsolved_event

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEBRANCHED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODEBRANCHED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.NODEBRANCHED:
            self.on_nodebranched_event()
        elif event.getType() == SCIP_EVENTTYPE.LPSOLVED:
            self.on_lpsolved_event()
        else:
            raise ValueError('event error')

# eventhdlr = MyEvent()
# model.includeEventhdlr(eventhdlr, "TestFirstLPevent", "python event handler to catch FIRSTLPEVENT")

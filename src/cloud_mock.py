class CloudServer:
    def __init__(self):
        self.valid_identifiers = {}  # service_id -> set of (mac, uid) allowed
        self.rotation_schedules = {} # device_id -> schedule

    def register_device(self, device_id, service_id, mac, uid, rotation_schedule):
        if service_id not in self.valid_identifiers:
            self.valid_identifiers[service_id] = set()
        self.valid_identifiers[service_id].add((mac, uid))
        self.rotation_schedules[device_id] = rotation_schedule

    def is_valid(self, service_id, mac, uid):
        if service_id not in self.valid_identifiers:
            return False
        return (mac, uid) in self.valid_identifiers[service_id]

    def report_edge_insight(self, report):
        """Receive enriched data from edge."""
        # In a real system, could update models or trigger actions
        pass
def write_csv_log(logs, filename=None):
    if filename is None:
        filename = 'log.csv'
    with open(filename, "w", encoding="utf-8") as f:
        f.write('\t'.join(('current_time', 'queue_length', 'channels_in_use', 'event_type', 'task_uid',
                           'time_arrived', 'time_entered_queue', 'processing_started', 'processing_ends', 'success\n')))
        for r in logs:
            t = r.task
            f.write("\t".join(
                (str(x) for x in (r.current_time, r.queue_length, r.channels_in_use, r.event_type, t.uid, t.time_arrived,
                                  t.time_entered_queue, t.time_processing_started, t.time_processing_ends, t.success
                                  ))
            ))
            f.write("\n")

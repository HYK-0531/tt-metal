semaphore_offset_index = 0


def get_next_semaphores(remote_semaphore_handles, semaphores_used):
    global semaphore_offset_index
    print("get_next_semaphores called with semaphore_offset_index:", semaphore_offset_index)
    start = semaphore_offset_index % len(remote_semaphore_handles)
    end = (start + semaphores_used) % len(remote_semaphore_handles)

    if start < end:
        semaphores = remote_semaphore_handles[start:end]
    else:
        # Wrap-around case
        semaphores = remote_semaphore_handles[start:] + remote_semaphore_handles[:end]

    semaphore_offset_index += semaphores_used
    return semaphores

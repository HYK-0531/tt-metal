def get_next_semaphores(remote_semaphore_handles, semaphore_offset_index, semaphores_used):
    start = semaphore_offset_index % len(remote_semaphore_handles)
    end = (start + semaphores_used) % len(remote_semaphore_handles)

    if start < end:
        semaphores = remote_semaphore_handles[start:end]
    else:
        # Wrap-around case
        semaphores = remote_semaphore_handles[start:] + remote_semaphore_handles[:end]

    semaphore_offset_index += semaphores_used
    return semaphores

# first line: 34
def compute_responses(
    network: "flyvis.network.CheckpointedNetwork",
    dataset_class: type,
    dataset_config: Dict,
    batch_size: int,
    t_pre: float,
    t_fade_in: float,
    cell_index: Optional[np.ndarray | str] = "central",
) -> xr.Dataset:
    """Compute responses and return.

    This function is compatible with joblib caching.
    """
    # Reconstruct the network
    network.recover()
    network = network.network  # type: flyvis.Network

    # Initialize dataset
    dataset = dataset_class(**dataset_config)

    if cell_index == "central":
        # Prepare central cells index
        cell_index = network.connectome.central_cells_index[:]

    stimuli_array = []
    responses_array = []

    # Compute responses
    for _, (stimulus, responses) in enumerate(
        network.stimulus_response(
            dataset,
            dataset.dt,
            t_pre=t_pre,
            t_fade_in=t_fade_in,
            batch_size=batch_size,
        )
    ):
        if cell_index is not None:
            responses = np.take(responses, cell_index, axis=-1)

        stimuli_array.append(stimulus)
        responses_array.append(responses)

    stimuli_array = np.concatenate(stimuli_array, axis=0)
    responses_array = np.concatenate(responses_array, axis=0)[None]

    return xr.Dataset(
        {
            'stimulus': (
                [
                    'sample',
                    'frame',
                    'channel',
                    'hex_pixel',
                ],
                stimuli_array,
            ),
            'responses': (
                [
                    'network_id',
                    'sample',
                    'frame',
                    'neuron',
                ],
                responses_array,
            ),
        },
        coords={
            'sample': np.arange(len(dataset)),
            **{
                col: ('sample', dataset.arg_df[col].values)
                for col in dataset.arg_df.columns
            },
        },
    )

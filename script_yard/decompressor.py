#!/usr/bin/env python3
"""
Decompress MCAP ROS2 bag files with zstd compression.
Requires: pip install mcap mcap-ros2-support zstandard
"""

import os
from pathlib import Path
from mcap.reader import make_reader
from mcap.writer import Writer


def decompress_mcap_bag(input_folder, output_folder=None):
    """
    Decompress MCAP ROS2 bag files from a folder.

    Args:
        input_folder (str): Path to folder containing compressed MCAP files
        output_folder (str): Path to output folder (default: input_folder + '_decompressed')

    Returns:
        list: Paths to decompressed MCAP files
    """
    input_path = Path(input_folder)

    if output_folder is None:
        output_folder = str(input_path) + '_decompressed'

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all MCAP files
    mcap_files = sorted(input_path.glob('*.mcap'))

    if not mcap_files:
        raise ValueError(f"No MCAP files found in {input_folder}")

    decompressed_files = []

    for mcap_file in mcap_files:
        print(f"Decompressing: {mcap_file.name}")

        output_file = output_path / mcap_file.name.replace('_compressed', '_decompressed')

        # Read compressed MCAP and write uncompressed
        with open(mcap_file, 'rb') as infile:
            reader = make_reader(infile)

            with open(output_file, 'wb') as outfile:
                writer = Writer(outfile)
                writer.start()

                # Track registered schemas and channels
                schema_map = {}  # old_id -> new_id
                channel_map = {}  # old_id -> new_id

                # First pass: register all schemas
                summary = reader.get_summary()
                if summary and summary.schemas:
                    for schema_id, schema in summary.schemas.items():
                        new_schema_id = writer.register_schema(
                            name=schema.name,
                            encoding=schema.encoding,
                            data=schema.data
                        )
                        schema_map[schema_id] = new_schema_id

                # Second pass: register all channels
                if summary and summary.channels:
                    for channel_id, channel in summary.channels.items():
                        new_channel_id = writer.register_channel(
                            topic=channel.topic,
                            message_encoding=channel.message_encoding,
                            schema_id=schema_map.get(channel.schema_id, channel.schema_id),
                            metadata=channel.metadata
                        )
                        channel_map[channel_id] = new_channel_id

                # Third pass: write all messages
                for schema, channel, message in reader.iter_messages():
                    writer.add_message(
                        channel_id=channel_map.get(channel.id, channel.id),
                        log_time=message.log_time,
                        data=message.data,
                        publish_time=message.publish_time
                    )

                writer.finish()

        decompressed_files.append(str(output_file))
        print(f"  -> Saved to: {output_file.name}")

    print(f"\nDecompressed {len(decompressed_files)} files to: {output_folder}")
    return decompressed_files





if __name__ == '__main__':
    import sys



    input_folder = "/home/aaron/uni/thesis/data/seq4/seq4_telemax"
    output_folder = "/home/aaron/uni/thesis/data/seq4/seq4_telemax_decompressed"

    try:
        decompress_mcap_bag(input_folder, output_folder)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def _fetch_and_export_body_annotations(cfg):
    dvid_node = (cfg['snapshot']['server'], cfg['snapshot']['uuid'])
    snapshot_tag = cfg['snapshot-tag']

    ann = fetch_body_annotations(*dvid_node)
    feather.write_feather(
        ann.reset_index().drop(columns=['json']),
        f'tables/body-annotations-{snapshot_tag}.feather'
    )

    # The result includes the original json as an extra column,
    # but that's not necessary for anything in this code.
    del ann['json']

    vc = ann['status'].value_counts().sort_index(ascending=False)
    vc = vc[vc > 0]
    vc = vc[vc.index != ""]
    vc.to_csv(f'tables/status-counts-{snapshot_tag}.csv', index=True, header=True)

    title = f'body status counts ({snapshot_tag})'
    p = vc.hvplot.barh(flip_yaxis=True, title=title)
    export_bokeh(
        hv.render(p),
        f"body-status-counts-{snapshot_tag}.html",
        title
    )
    return ann

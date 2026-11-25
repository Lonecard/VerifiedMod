#!/usr/bin/env bash
# Downloads vanilla Minecraft 1.21 block/item textures (from the `minecraft-assets`
# npm package) into ./mc_assets. They are Mojang's assets, so they stay out of git.
set -euo pipefail
cd "$(dirname "$0")"
VER=1.21.11
TMP=$(mktemp -d)
(cd "$TMP" && npm pack minecraft-assets@1.19.0 --silent >/dev/null && tar xzf minecraft-assets-*.tgz \
    "package/minecraft-assets/data/$VER/blocks" "package/minecraft-assets/data/$VER/items")
mkdir -p mc_assets/block mc_assets/item
for b in stone andesite tuff deepslate cobbled_deepslate gravel diamond_ore deepslate_diamond_ore \
         lava_still torch pointed_dripstone_down_tip pointed_dripstone_down_frustum pointed_dripstone_down_base \
         dripstone_block calcite smooth_basalt; do
  cp "$TMP/package/minecraft-assets/data/$VER/blocks/$b.png" mc_assets/block/
done
cp "$TMP/package/minecraft-assets/data/$VER/items/diamond_pickaxe.png" mc_assets/item/
rm -rf "$TMP"
echo "textures saved to $(pwd)/mc_assets"

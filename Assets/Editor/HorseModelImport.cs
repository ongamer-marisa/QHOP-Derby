using UnityEditor;

public sealed class HorseModelImport : AssetPostprocessor
{
    void OnPreprocessModel()
    {
        if(!assetPath.EndsWith("/HorseGame.fbx"))return;
        var importer=(ModelImporter)assetImporter;
        importer.animationType=ModelImporterAnimationType.Generic;
        importer.importAnimation=false;
        importer.isReadable=true;
        importer.importBlendShapes=true;
        importer.importNormals=ModelImporterNormals.Import;
    }
}

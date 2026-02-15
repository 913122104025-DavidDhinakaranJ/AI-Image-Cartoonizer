import { ChangeEvent } from "react";

interface ImageUploaderProps {
  onSelectFile: (file: File | null) => void;
}

export function ImageUploader({ onSelectFile }: ImageUploaderProps) {
  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    onSelectFile(file);
  };

  return (
    <label className="panel uploader">
      <span className="label">Upload Image</span>
      <input
        type="file"
        accept=".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp"
        onChange={onChange}
      />
      <p className="hint">Accepted: JPG, PNG, WEBP</p>
    </label>
  );
}

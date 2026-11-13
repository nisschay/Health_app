import { ChartNoAxesCombined } from "lucide-react";

/** The one LabTrail logo: mark plus name. */
export default function Brand() {
  return (
    <>
      <span className="brand-mark" aria-hidden="true">
        <ChartNoAxesCombined size={22} />
      </span>
      <span className="brand-name">LabTrail</span>
    </>
  );
}
